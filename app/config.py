# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

from omegaconf import OmegaConf

from definitions import ROOT_DIR

path = os.path.join(ROOT_DIR, 'config.yaml')
config = OmegaConf.load(path)
