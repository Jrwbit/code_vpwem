"""Registered full-gate epoch selection and retention of the current candidate.

These additional development constraints never relax original V4 comparators.
They compare the same fixed frame denominators and accepted predictions.
"""


def current_retention_gate(primary,current):
    p=primary['accepted'];b=current['accepted'];s=primary['sequence'];old=current['sequence']
    floors={role:max(0.,(value['recall'] or 0)-.05) for role,value in current['roles'].items() if value['visible']>=10}
    checks=dict(correct_boxes_retained=p['correct']>=b['correct'],acquisition_retained=p['wrong_acquire']<=b['wrong_acquire'],
        carry_retained=p['wrong_carry']<=b['wrong_carry'],
        stable_class_rate_retained=(s['stable_class_switch_rate'] or 0)<=(old['stable_class_switch_rate'] or 0),
        correct_at_change_retained=s['correct_box_at_change']>=old['correct_box_at_change'],
        stale_class_retained=s['stale_previous_class_at_change']<=old['stale_previous_class_at_change'],
        role_recall_retained=all((primary['roles'][role]['recall'] or 0)>=floor for role,floor in floors.items()))
    return dict(checks,joint_goal_met=all(checks.values()),role_recall_floors=floors)


def checkpoint_merit(value):
    eligible=value['joint_growth_goal_met'];c=value['calibration'];m=c['metrics'];s=c['sequence']
    return (eligible,m['recall'] if eligible else m['f1'],-m['wrong_acquire'],-m['wrong_carry'],
        -(s['stable_class_switch_rate'] or 0),s['correct_box_at_change'],m['precision'])
