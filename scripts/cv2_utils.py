#!/usr/bin/env python

import dict_utils as du
import copy, cv2, pickle

class Config:
    def __init__(self, file, initialConfig = None):
        self.filename = file

        if initialConfig is None:
            self.load()
        else:
            self._config = initialConfig

    def sliders(self, windowName, subConfig=[]):
        self._cfgSlidersHelper(windowName, subConfig)

    def save(self):
        with open(self.filename, 'wb') as fb:
            pickle.dump(self._config, fb)

    def load(self):
        with open(self.filename, 'rb') as fb:
            self._config = pickle.load(fb)

    def get(self, path=[], keys=None):
        val = du.getFromDict(self._config, path)

        if keys is not None:
            val = map(val.get, keys)

        return val

    def _cfgSlidersHelper(self, windowName, path):
        for key, value in du.getFromDict(self._config, path).items():
            tmpPath = copy.deepcopy(path)
            tmpPath.append(key)

            if isinstance(value, dict):
                self._cfgSlidersHelper(windowName, tmpPath)

            else:
                cv2.createTrackbar(
                    " ".join(tmpPath),
                    windowName,
                    value,
                    255,
                    self._setterLambda(self._config, tmpPath)
                )

    def _setterLambda(self, cfg, path):
        return lambda val: du.setInDict(self._config, path, val)