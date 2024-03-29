# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from densepose.data.video import FirstKFramesSelector, LastKFramesSelector, RandomKFramesSelector
import secrets


class TestFrameSelector(unittest.TestCase):
    def test_frame_selector_random_k_1(self):
        _SEED = 43
        _K = 4
        secrets.SystemRandom().seed(_SEED)
        selector = RandomKFramesSelector(_K)
        frame_tss = list(range(0, 20, 2))
        _SELECTED_GT = [0, 8, 4, 6]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_random_k_2(self):
        _SEED = 43
        _K = 10
        secrets.SystemRandom().seed(_SEED)
        selector = RandomKFramesSelector(_K)
        frame_tss = list(range(0, 6, 2))
        _SELECTED_GT = [0, 2, 4]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_first_k_1(self):
        _K = 4
        selector = FirstKFramesSelector(_K)
        frame_tss = list(range(0, 20, 2))
        _SELECTED_GT = frame_tss[:_K]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_first_k_2(self):
        _K = 10
        selector = FirstKFramesSelector(_K)
        frame_tss = list(range(0, 6, 2))
        _SELECTED_GT = frame_tss[:_K]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_last_k_1(self):
        _K = 4
        selector = LastKFramesSelector(_K)
        frame_tss = list(range(0, 20, 2))
        _SELECTED_GT = frame_tss[-_K:]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)

    def test_frame_selector_last_k_2(self):
        _K = 10
        selector = LastKFramesSelector(_K)
        frame_tss = list(range(0, 6, 2))
        _SELECTED_GT = frame_tss[-_K:]
        selected = selector(frame_tss)
        self.assertEqual(_SELECTED_GT, selected)
