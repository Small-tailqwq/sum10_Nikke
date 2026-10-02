#!/usr/bin/env python3
"""Negative controls and invariants for the independent certificate checker."""
import copy
import json
from pathlib import Path
import unittest
from check_corpus import InvalidCertificate,replay

ROOT=Path(__file__).resolve().parent

class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.board={'rows':2,'cols':2,'values':[1,9,4,6],'initial_active':[1,1,1,1]}
        self.path=[[0,0,0,1],[1,0,1,1]]
    def test_legal_full_clear(self):
        result=replay(self.board,self.path)
        self.assertEqual(result['score'],4)
        self.assertEqual(result['remaining_indices'],[])
        self.assertEqual(result['unanchored_moves'],0)
    def test_partial_result_allowed_only_when_explicit(self):
        with self.assertRaises(InvalidCertificate): replay(self.board,self.path[:1])
        result=replay(self.board,self.path[:1],require_full=False)
        self.assertEqual(result['score'],2)
        self.assertEqual(result['remaining_indices'],[2,3])
    def test_wrong_sum_rejected(self):
        with self.assertRaises(InvalidCertificate): replay(self.board,[[0,0,1,1]])
    def test_duplicate_move_rejected(self):
        with self.assertRaises(InvalidCertificate): replay(self.board,[self.path[0],self.path[0]])
    def test_bounds_rejected(self):
        for rectangle in ([-1,0,0,1],[0,0,2,1],[1,0,0,1],[0,1,0,0]):
            with self.subTest(rectangle=rectangle),self.assertRaises(InvalidCertificate):
                replay(self.board,[rectangle])
    def test_noninteger_coordinates_rejected(self):
        for coordinate in (False,0.0,'0'):
            with self.subTest(coordinate=coordinate),self.assertRaises(InvalidCertificate):
                replay(self.board,[[coordinate,0,0,1]])
    def test_digit_domain_rejected(self):
        for value in (0,10,True,1.0):
            board=copy.deepcopy(self.board);board['values'][0]=value
            with self.subTest(value=value),self.assertRaises(InvalidCertificate): replay(board,self.path)
    def test_missing_or_extra_value_rejected(self):
        for vals in ([1,9,4],[1,9,4,6,2]):
            board=copy.deepcopy(self.board);board['values']=vals
            with self.subTest(vals=vals),self.assertRaises(InvalidCertificate): replay(board,self.path)
    def test_falsified_count_rejected(self):
        with self.assertRaises(InvalidCertificate): replay(self.board,self.path,expected_counts=[3,1])
    def test_nondense_corpus_rejected(self):
        self.board['initial_active'][0]=0
        with self.assertRaises(InvalidCertificate): replay(self.board,self.path)
    def test_grid_mismatch_rejected(self):
        self.board['grid']=['11','46']
        with self.assertRaises(InvalidCertificate): replay(self.board,self.path)
    def test_development_long_range_and_anchor_coverage(self):
        boards=json.loads((ROOT/'development'/'boards.json').read_text())
        certs=json.loads((ROOT/'development'/'certificates.json').read_text())
        unsupported=0
        for board,cert in zip(boards,certs):
            result=replay(board,cert['path'],expected_counts=cert['removed_counts'])
            self.assertEqual(result['score'],160)
            self.assertGreater(result['dependent_moves'],0)
            self.assertGreater(result['long_range_moves'],0)
            self.assertGreater(result['max_dependency_depth'],1)
            self.assertEqual(set(result['digit_counts']),set(range(1,10)))
            if board['family']!='layered_anchors': self.assertEqual(result['unanchored_moves'],0)
            unsupported+=result['unanchored_moves']
        self.assertGreater(unsupported,0)

if __name__=='__main__':unittest.main(verbosity=2)
