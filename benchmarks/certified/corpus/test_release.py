#!/usr/bin/env python3
"""Exercise release on dummy fixtures only: never release actual holdout in tests."""
import json
from pathlib import Path
import tempfile
import unittest
from release_holdout import release,sha_json

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.fixture=self.root/'fixture'
        (self.fixture/'sealed_holdout').mkdir(parents=True)
        boards=[{'id':'dummy','values':[1,9]}]
        (self.fixture/'sealed_holdout'/'boards.json').write_text(json.dumps(boards))
        (self.fixture/'sealed_holdout'/'certificates.json').write_text('DO NOT EXPORT')
        (self.fixture/'sealed_holdout'/'seeds.json').write_text('DO NOT EXPORT')
        (self.fixture/'manifest.json').write_text(json.dumps({'splits':{'sealed_holdout':{'boards_sha256':sha_json(boards)}}}))
        self.candidate=self.root/'candidate.py';self.candidate.write_text('# candidate')
        self.config=self.root/'config.json';self.config.write_text('{"budget":1}')
        self.destination=self.root/'evaluation'
    def run_release(self):
        return release([self.candidate],self.config,self.destination,self.fixture,False)
    def test_frozen_candidate_and_settings(self):
        result=self.run_release()
        self.assertEqual(result['configuration'],{'budget':1})
        self.assertEqual(len(result['candidate_files'][0]['sha256']),64)
        self.assertTrue(result['frozen_at_utc'].endswith('+00:00'))
    def test_only_boards_and_freeze_record_exported(self):
        self.run_release()
        self.assertEqual({p.name for p in self.destination.iterdir()},
                         {'candidate_freeze.json','primary_holdout_boards.json'})
    def test_existing_destination_rejected(self):
        self.run_release()
        with self.assertRaises(ValueError):self.run_release()
    def test_mutated_holdout_rejected_before_export(self):
        (self.fixture/'sealed_holdout'/'boards.json').write_text('[]')
        with self.assertRaises(ValueError):self.run_release()
        self.assertFalse(self.destination.exists())
    def test_empty_candidate_list_rejected(self):
        with self.assertRaises(ValueError):release([],self.config,self.destination,self.fixture,False)
        self.assertFalse(self.destination.exists())

if __name__=='__main__':unittest.main(verbosity=2)
