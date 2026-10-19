import itertools
import unittest

import numpy as np

from cnn_analysis.cross_surface import SEEDS, combinations, assemble, split_records
from cnn_analysis.quality import EXCLUDED, position_group
from cnn_analysis.surface_audit import surface


class CrossSurfaceTests(unittest.TestCase):
    def records(self):
        return [dict(sample_id=f'{s}-E01-R00-P{p:02}-H{h:03}', stick_id=s,
                     impact_point_id=p, strike_id=h)
                for s,p,h in itertools.product(['WOOD01','WOOD02'], [1,2,3], range(1,51))]

    def test_surface_boundaries_and_counts(self):
        rows=self.records()
        self.assertEqual(sum(surface(r['stick_id'],r['impact_point_id'],r['strike_id'])=='carpet' for r in rows),129)
        for s,p,h,expected in [('WOOD01',1,45,'table'),('WOOD01',1,46,'carpet'),
                               ('WOOD01',2,26,'table'),('WOOD01',2,27,'carpet'),
                               ('WOOD01',3,25,'carpet'),('WOOD01',3,26,'table'),
                               ('WOOD02',2,25,'table'),('WOOD02',2,26,'carpet')]:
            self.assertEqual(surface(s,p,h),expected)

    def test_independent_source_splits_and_finite_calibration(self):
        for support,seed,quality in itertools.product(['carpet','table'],SEEDS,[False,True]):
            rows=[r for r in self.records() if r['sample_id'] not in EXCLUDED and surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==support]
            parts=split_records(rows,seed,quality)
            pools=[{r['sample_id'] for r in rr} for rr in parts.values()]
            self.assertTrue(all(not(a&b) for a,b in itertools.combinations(pools,2)))
            self.assertEqual(set.union(*pools),{r['sample_id'] for r in rows})
            self.assertEqual(parts,split_records(rows,seed,quality))
            if quality:
                for group in ['middle','endpoint']:
                    self.assertGreaterEqual(sum(position_group(r['impact_point_id'])==group for r in parts['calibration']),19)

    def test_triplets_positions_no_cross_wood_or_cross_surface_and_disjoint(self):
        records={r['sample_id']:r for r in self.records()}
        cache={sid:np.full(12,r['impact_point_id'],dtype=np.float32) for sid,r in records.items()}
        for support in ['carpet','table']:
            rows=[r for r in records.values() if r['sample_id'] not in EXCLUDED and surface(r['stick_id'],r['impact_point_id'],r['strike_id'])==support]
            for disjoint in [True,False]:
                triples=combinations(rows,1000,191019,disjoint)
                x,y=assemble(triples,records,cache)
                self.assertTrue(np.all(x[:,:,0]==1) and np.all(x[:,:,1]==2) and np.all(x[:,:,2]==3))
                self.assertEqual(set(y.tolist()),{0,1})
                self.assertEqual(len(set(triples)),len(triples))
                used=[sid for triple in triples for sid in triple]
                self.assertFalse(set(used)&EXCLUDED)
                if disjoint:self.assertEqual(len(used),len(set(used)))
            bad=[(triples[0][0],triples[0][0],triples[0][2])]
            with self.assertRaises(AssertionError):assemble(bad,records,cache)


if __name__=='__main__':unittest.main()
