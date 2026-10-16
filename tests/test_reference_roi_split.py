import unittest

from util.data_split import build_group_split_indices


class ReferenceROISplitTest(unittest.TestCase):
    def test_filtering_preserves_original_membership(self):
        groups = [f'roi{i}' for i in range(20) for _ in range(2)]
        train, valid = build_group_split_indices(groups, 0.8, 42)
        train_rois = {groups[i] for i in train}
        valid_rois = {groups[i] for i in valid}
        removed = {groups[train[0]], groups[valid[0]]}
        filtered = [g for g in groups if g not in removed]
        ft, fv = build_group_split_indices(filtered, 0.8, 42, groups)
        self.assertEqual({filtered[i] for i in ft}, train_rois - removed)
        self.assertEqual({filtered[i] for i in fv}, valid_rois - removed)

    def test_unknown_roi_is_rejected(self):
        with self.assertRaises(ValueError):
            build_group_split_indices(['unknown'], 0.8, 42, ['roi1', 'roi2'])


if __name__ == '__main__':
    unittest.main()
