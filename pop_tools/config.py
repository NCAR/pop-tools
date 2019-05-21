"""Configuration for pop-tools"""

import os
from subprocess import Popen, PIPE

import jinja2
import yaml

package_dir = os.path.dirname(__file__)

inputdata_repo = 'https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata'
inputdata_local = '/glade/p/cesmdata/cseg/inputdata'

if os.path.exists(inputdata_local):
    INPUTDATA = inputdata_local
else:
    scratch = os.path.join(os.path.expanduser('~'), 'scratch')
    INPUTDATA = os.path.join(scratch, 'inputdata')

INPUT_TEMPLATES = os.path.join(package_dir, 'input_templates')
grid_def_file = os.path.join(package_dir, 'pop_grid_definitions.yaml')


def gen_grid_defs(grid_def_file):
    """Read grid pop_grid_definitions file."""
    with open(grid_def_file) as f:
        grid_defs = yaml.safe_load(f)

    # replace template keys
    for grid, grid_attrs in grid_defs.items():
        for k, v in grid_attrs.items():
            if not isinstance(v, str):
                continue
            template = jinja2.Template(v)
            grid_attrs[k] = template.render(INPUTDATA=INPUTDATA,
                                            INPUT_TEMPLATES=INPUT_TEMPLATES)
    return grid_defs


def inputdata_relpath(file_fullpath):
    """Return part of path following /path/to/inputdata/"""
    fullpath_list = file_fullpath.split('/')
    ndx = fullpath_list.index('inputdata')
    return '/'.join(fullpath_list[ndx + 1:])


def svn_export(repo_path, local_path):
    """svn export"""

    p = Popen(['svn', 'export', repo_path, local_path])
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout)
        print(stderr)
        raise Exception('svn error')


def ensure_inputdata():
    """Checkout necessary files from inputdata."""

    if not os.path.exists(INPUTDATA):
        if not os.path.isdir(INPUTDATA):
            os.makedirs(INPUTDATA)

    indat_grid_file_keys = ['horiz_grid_fname', 'topography_fname',
                            'region_mask_fname']

    for grid, grid_attrs in grid_defs.items():

        for key, val in grid_attrs.items():

            if key in indat_grid_file_keys and not os.path.lexists(val):
                os.makedirs(os.path.dirname(val), exist_ok=True)

                repo_path = f'{inputdata_repo}/{inputdata_relpath(val)}'
                svn_export(repo_path, val)


grid_defs = gen_grid_defs(grid_def_file)
ensure_inputdata()
