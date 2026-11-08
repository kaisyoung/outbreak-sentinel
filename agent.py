"""
Online quarantine policy for a noisy, partially observed outbreak.

A factorial hidden Markov model: one SEIR chain per person, coupled through the
per-shift room-sharing graph. Rapid antigen, temperature and symptom readings
are emitted by the current status; PCR is emitted by the PREVIOUS shift's
status (the assay runs one shift behind).

Inference is structured mean-field: given the other chains' marginals each
chain is an ordinary 4-state HMM, solved exactly by forward-backward, and the
chains are reconciled by iterating. Decisions come from a one-step lookahead on
the belief-state MDP, since quarantine is an absorbing action.
"""

import collections
import json
import os
from itertools import product

STATES = ('S', 'E', 'I', 'R')

N_STAFF = 60             # 60 staffs
N_SHIFTS = 14            # 14 shifts

C_EXPOSURE = 40.0        # cost per shift of clearing a truly infectious person
C_QUARANTINE = 3.0       # cost per shift of holding someone in quarantine

RISK_FACTOR = 0.7        # <1 quarantines earlier/more, >1 later/less
LOOKAHEAD_HAZARD = 0.0   # S->E rate assumed when projecting the 'clear' branch

P_SENSOR_MISSING = 0.2

SMOOTHING_ITERATIONS = 1

DEBUG = False            # set True to write a per-decision trace to debug.log

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, 'parameter.json'), 'r') as f:
    parameter = json.load(f)

def _as_distribution(table, default=None):
    dist = {s: float(table.get(s, 0.0)) for s in STATES}
    total = sum(dist.values())
    if total <= 0.0:
        return dict(default) if default else {s: 0.25 for s in STATES}
    return {s: v / total for s, v in dist.items()}

def _build_transition(raw):
    matrix = {}
    for s in STATES:
        row = raw.get(s, {})
        row = {s2: float(row.get(s2, 0.0)) for s2 in STATES}
        total = sum(row.values())
        if total <= 0.0:
            row = {s2: (1.0 if s2 == s else 0.0) for s2 in STATES}
        else:
            row = {s2: v / total for s2, v in row.items()}
        matrix[s] = row
    return matrix

def _fit_contact_beta(table, fallback=0.08):
    estimates = []
    for k_str, p in table.items():
        k = int(k_str)
        if k <= 0 or not (0.0 < p < 1.0):
            continue
        estimates.append(1.0 - (1.0 - p) ** (1.0 / k))
    if not estimates:
        return fallback
    return sum(estimates) / len(estimates)

# --- Learned parameters ---------------------------------------------------

PRIOR = _as_distribution(parameter.get('prior', {}))
TRANSITION = _build_transition(parameter.get('transition', {}))
CONTACT_BETA = parameter.get('contact_beta')
if CONTACT_BETA is None:
    CONTACT_BETA = _fit_contact_beta(parameter.get('contact_infection', {}))

SENSORS = ('rapid', 'temp', 'symptom')

PCR_TABLE = parameter.get('pcr_delay') or parameter.get('pcr', {})

state = {}
for staff_id in range(N_STAFF):
    state[staff_id] = {
        'belief': dict(PRIOR),   # smoothed belief over X_i at the current shift
        'history': {},           # shift -> that shift's observation record
        'traj': {},              # shift -> smoothed belief over X_i(shift)
    }

room_index = {}
contact_cache = {}

if DEBUG:
    import logging
    logging.basicConfig(filename='debug.log', filemode='w',
                        level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)

def normalise(belief, fallback=None):
    total = sum(belief.values())
    if total <= 0.0:
        return dict(fallback) if fallback else dict(PRIOR)
    return {s: v / total for s, v in belief.items()}

def update_history(staff_state, observation, shift):
    staff_state['history'][shift] = {
        'rooms': observation['rooms'],
        'rapid': observation['rapid'],
        'pcr': observation['pcr'],
        'temp': observation['temp'],
        'symptom': observation['symptom'],
    }

def build_room_index(shift):
    if shift in room_index:
        return room_index[shift]
    index = collections.defaultdict(list)
    for staff_id, staff_state in state.items():
        record = staff_state['history'].get(shift)
        if record:
            for room in record['rooms']:
                index[room].append(staff_id)
    room_index[shift] = index
    return index

def contacts_at(staff_id, shift):
    key = (staff_id, shift)
    cached = contact_cache.get(key)
    if cached is not None:
        return cached

    record = state[staff_id]['history'].get(shift)
    if not record:
        contact_cache[key] = ()
        return ()

    index = room_index.get(shift, {})
    contacts = set()
    for room in record['rooms']:
        contacts.update(index.get(room, ()))
    contacts.discard(staff_id)

    contacts = tuple(contacts)
    contact_cache[key] = contacts
    return contacts

# Evidence, dynamics

def evidence_likelihood(staff_id, shift):
    likelihood = {s: 1.0 for s in STATES}

    record = state[staff_id]['history'].get(shift)
    if record is None:
        return likelihood

    for sensor in SENSORS:
        value = record[sensor]
        if value is None:
            continue
        table = parameter.get(sensor)
        if table is None:
            continue
        for s in STATES:
            p = table.get(s, {}).get(value)
            if p is not None:
                likelihood[s] *= p

    later = state[staff_id]['history'].get(shift + 1)
    if later is not None and later['pcr'] is not None:
        for s in STATES:
            p = PCR_TABLE.get(s, {}).get(later['pcr'])
            if p is not None:
                likelihood[s] *= p

    return likelihood

def update_contact_probability(staff_id, shift, p_infectious):
    escape = 1.0
    for other in contacts_at(staff_id, shift - 1):
        escape *= (1.0 - CONTACT_BETA * p_infectious[other].get(shift - 1, 0.0))
    return 1.0 - escape

def transition_row(status, hazard=None):
    if status == 'S' and hazard is not None:
        return {'S': 1.0 - hazard, 'E': hazard, 'I': 0.0, 'R': 0.0}
    return TRANSITION[status]

def update_transition_probability(belief, hazard=None):
    predicted = {s: 0.0 for s in STATES}
    for s in STATES:
        mass = belief[s]
        if mass <= 0.0:
            continue
        for s2, p in transition_row(s, hazard).items():
            predicted[s2] += mass * p
    return normalise(predicted, fallback=belief)

# Inference: mean-field smoothing over the factorial HMM

def chain_smooth(staff_id, last_shift, p_infectious):
    shifts = range(last_shift + 1)
    likelihood = {s: evidence_likelihood(staff_id, s) for s in shifts}
    hazard = {s: update_contact_probability(staff_id, s, p_infectious)
              for s in shifts if s > 0}

    alpha = {0: normalise({s: PRIOR[s] * likelihood[0][s] for s in STATES},
                          fallback=PRIOR)}
    for shift in range(1, last_shift + 1):
        predicted = update_transition_probability(alpha[shift - 1],
                                                  hazard[shift])
        alpha[shift] = normalise(
            {s: predicted[s] * likelihood[shift][s] for s in STATES},
            fallback=predicted)

    beta = {last_shift: {s: 1.0 for s in STATES}}
    for shift in range(last_shift, 0, -1):
        nxt, lik = beta[shift], likelihood[shift]
        message = {}
        for s in STATES:
            row = transition_row(s, hazard[shift])
            message[s] = sum(p * lik[s2] * nxt[s2] for s2, p in row.items())
        beta[shift - 1] = normalise(message,
                                    fallback={s: 1.0 for s in STATES})

    return {shift: normalise({s: alpha[shift][s] * beta[shift][s]
                              for s in STATES}, fallback=alpha[shift])
            for shift in shifts}

def smooth(current_shift, iterations=None):
    if iterations is None:
        iterations = SMOOTHING_ITERATIONS

    for _ in range(iterations):
        p_infectious = {sid: {sh: b['I'] for sh, b in s['traj'].items()}
                        for sid, s in state.items()}

        for staff_id, staff_state in state.items():
            history = staff_state['history']
            if 0 not in history:
                continue
            last_shift = max(s for s in range(current_shift + 1)
                             if s in history)

            traj = chain_smooth(staff_id, last_shift, p_infectious)
            staff_state['traj'] = traj
            staff_state['belief'] = traj[last_shift]

# Decision: one-step lookahead on the belief-state MDP 

def _build_clear_cost_table():
    table = [None] * (N_SHIFTS + 1)
    table[N_SHIFTS] = {s: 0.0 for s in STATES}
    for shift in range(N_SHIFTS - 1, -1, -1):
        nxt = table[shift + 1]
        table[shift] = {
            s: (C_EXPOSURE if s == 'I' else 0.0)
               + sum(p * nxt[s2]
                     for s2, p in transition_row(s, LOOKAHEAD_HAZARD).items())
            for s in STATES}
    return table

CLEAR_COST = _build_clear_cost_table()

def _build_observation_outcomes():
    outcomes = []
    per_sensor = []
    for sensor in SENSORS:
        table = parameter.get(sensor, {})
        values = sorted({v for s in STATES for v in table.get(s, {})})
        choices = [(None, {s: P_SENSOR_MISSING for s in STATES})]
        for value in values:
            choices.append((value, {
                s: (1.0 - P_SENSOR_MISSING) * table.get(s, {}).get(value, 0.0)
                for s in STATES}))
        per_sensor.append(choices)

    for combo in product(*per_sensor):
        likelihood = {s: 1.0 for s in STATES}
        for _, factor in combo:
            for s in STATES:
                likelihood[s] *= factor[s]
        outcomes.append(likelihood)
    return outcomes

OBSERVATION_OUTCOMES = _build_observation_outcomes()

def expected_clear_cost(belief, shift):
    if shift >= N_SHIFTS:
        return 0.0
    costs = CLEAR_COST[shift]
    return sum(belief[s] * costs[s] for s in STATES)

def lookahead_clear_cost(belief, shift):
    exposure_now = C_EXPOSURE * belief['I']
    if shift + 1 >= N_SHIFTS:
        return exposure_now

    predicted = update_transition_probability(belief, hazard=LOOKAHEAD_HAZARD)
    quarantine_next = RISK_FACTOR * C_QUARANTINE * (N_SHIFTS - shift - 1)
    costs = CLEAR_COST[shift + 1]

    continuation = 0.0
    for likelihood in OBSERVATION_OUTCOMES:
        joint = {s: predicted[s] * likelihood[s] for s in STATES}
        evidence = sum(joint.values())
        if evidence <= 0.0:
            continue
        clear_next = sum(joint[s] * costs[s] for s in STATES) / evidence
        continuation += evidence * min(quarantine_next, clear_next)

    return exposure_now + continuation

def should_quarantine(belief, shift):
    quarantine_cost = RISK_FACTOR * C_QUARANTINE * max(0, N_SHIFTS - shift)

    if expected_clear_cost(belief, shift) <= quarantine_cost:
        return False                      # even never quarantining is cheaper
    if C_EXPOSURE * belief['I'] >= quarantine_cost:
        return True                       # one shift of exposure already costs more

    return lookahead_clear_cost(belief, shift) > quarantine_cost

def print_staff_debug(shift, staff_id, observation, belief_before,
                      belief_after, action):
    if not DEBUG:
        return
    logger.info('=' * 70)
    logger.info(f'Shift {shift}  Staff {staff_id}')
    logger.info(f"  rooms={observation['rooms']} pcr={observation['pcr']} "
                f"rapid={observation['rapid']} temp={observation['temp']} "
                f"symptom={observation['symptom']}")
    logger.info('  before: ' + ' '.join(f'{s}={belief_before[s]:.3f}'
                                        for s in STATES))
    logger.info('  after : ' + ' '.join(f'{s}={belief_after[s]:.3f}'
                                        for s in STATES))
    logger.info(f'  decision: {action}')

# Entry point

def get_action(observations):
    """One shift: record the contact graph, re-infer every trajectory from all
    evidence so far, then decide.  Returns 'clear'/'quarantine' per staff id."""

    global state

    actions = {}
    shift = observations['shift']
    on_duty = observations['staff']

    for staff_id, obs in on_duty.items():
        if staff_id not in state:
            state[staff_id] = {'belief': dict(PRIOR), 'history': {}, 'traj': {}}
        update_history(state[staff_id], obs, shift)
    build_room_index(shift)

    belief_before = {sid: dict(state[sid]['belief']) for sid in on_duty}
    smooth(shift)

    for staff_id, obs in on_duty.items():
        belief = state[staff_id]['belief']
        action = 'quarantine' if should_quarantine(belief, shift) else 'clear'
        actions[staff_id] = action

        print_staff_debug(shift, staff_id, obs, belief_before[staff_id],
                          belief, action)

    return actions
