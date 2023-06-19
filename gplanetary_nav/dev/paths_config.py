#!/usr/bin/env python

""" 
    Paths configuration for the development of the gplanetary_nav package

    Author: Olivier Lamarre
    Affl.: STARS Laboratory, University of Toronto
"""

from pathlib import Path
import logging
import yaml

log = logging.getLogger(__name__)

# Absolute path to project repo
REPO_PATH = Path(__file__).resolve().parents[2]

# Path to the package's yaml config file
CFG_YML_PATH = Path(REPO_PATH, 'config/paths_config.yaml')

DEFAULT_CFG = {
    'data_dir': Path(REPO_PATH, 'data'),
    'logs_dir': Path(REPO_PATH, 'logs'),
    'params_dir': Path(REPO_PATH, 'params'),
    'config_dir': Path(REPO_PATH, 'config'),
    'tests_dir': Path(REPO_PATH, 'tests')
}

class _Config:

    def __init__(self):
        log.info('Loading config')

        try:
            with open(CFG_YML_PATH) as f:
                self.config = yaml.load(f, yaml.Loader)
        except FileNotFoundError:
            with open('../'+CFG_YML_PATH) as f:
                self.config = yaml.load(f, yaml.Loader)
        
        for k, default in DEFAULT_CFG.items():
            self.check_path_param(k, default)
        
        self.config['root'] = REPO_PATH
    
    def check_path_param(self, param_name, default):
        """Validity of a path parameter and set to default if none provided
        
        Args:
            param_name (str): path parameter name from the config file
            default (str): the default path parameter
        """

        # Set default path if not specified in yaml
        if (self.config[param_name] is None or 
            param_name not in self.config):
            log.warn(f"No '{param_name}' param, defaulting to: {default}")
            self.config[param_name] = default
        elif not Path(self.config[param_name]).exists():
            # Check that custom path exists, otherwise revert to default
            log.warn(f"Path does not exist: {self.config[param_name]} ,"
                     f"defaulting to {default}")
            self.config[param_name] = default
        else:
            log.info(f"'{param_name}' set to {self.config[param_name]}")

    def __getattr__(self, name):
        try:
            return self.config[name]
        except KeyError:
            log.error(f'Could not find configuration parameter {name}')


REPO_CFG = _Config()
