import copy
import unittest

from check_interrupted_return import completed_prefix


class CompletedPrefixTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'segments': [{'group': 'left_arm', 'steps': [
            {'target': [v]*7} for v in (.1, .2, .3)]}]}
        self.events = [{'event': 'route_step_done', 'step': i+1, 'group': 'left_arm',
                        'target': step['target']} for i, step in enumerate(self.plan['segments'][0]['steps'][:2])]

    def test_only_native_acknowledged_prefix_is_completed(self):
        self.assertEqual(completed_prefix(self.plan, [{'event': 'progress'}, *self.events]), 2)
        self.assertEqual(completed_prefix(self.plan, self.events[:1]), 1)

    def test_out_of_order_wrong_source_or_nonfinite_acknowledgements_are_rejected(self):
        for field, value in [('step', 3), ('step', True), ('group', 'right_arm'),
                             ('target', [9]*7), ('target', [float('nan')]*7), ('target', [0]*6)]:
            events = copy.deepcopy(self.events)
            events[1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                completed_prefix(self.plan, events)
        with self.assertRaises(ValueError):
            completed_prefix(self.plan, list(reversed(self.events)))

    def test_empty_or_already_complete_route_has_no_interrupted_prefix(self):
        for events in ([], [*self.events, {'event': 'route_step_done', 'step': 3,
                'group': 'left_arm', 'target': [.3]*7}]):
            with self.assertRaises(ValueError):
                completed_prefix(self.plan, events)


if __name__ == '__main__':
    unittest.main()
