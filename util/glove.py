from scipy.interpolate import interp1d
from collections import OrderedDict
from joblib import dump, load
from re import findall
import pandas as pd
import numpy as np
import json
import os
from warnings import warn

from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .bids import DataSink

_fingers = {
    '2': 'Index',
    '3': 'Middle',
    '4': 'Ring',
    '5': 'Pinky'
}
_knuckles = {
    'mcp': '2',
    'pm': '3',
    'md': '4'
}

def _map_to_leap(joint_nm):
    '''
    see mappings used in:
    https://github.com/seanschneeweiss/RoSeMotion/
        blob/e93644e4da39ef6d7bfffcc62358c7e68a38f7c6/
        app/AnyWriter.py#L129

    For whatever reason, the RoSeMotion methods paper
    says adbuctions are about Y-axis, but the software
    itself indicates they're about Z-axis instead.
    '''
    info = findall('(\w+)(\d)_(\w+)', joint_nm)[0]
    name = 'RightHand'
    name += _fingers[info[1]]
    name += _knuckles[info[0]]
    name += '_X' if info[2] == 'flexion' else '_Z'
    name += 'rotation'
    return name

_mapping = OrderedDict()

## thumb is special so we specify manually:
# flexion and adduction for CMC joint are mixed up in MyoSuite,
# https://github.com/MyoHub/myosuite/issues/112
# but aren't mixed up here -- they will be swapped in
# the `load_joint_angles` function if param `myo_fix = True`
_mapping['cmc_flexion'] = 'RightHandThumb2_Xrotation'
_mapping['cmc_abduction'] = 'RightHandThumb2_Zrotation'
_mapping['mp_flexion'] = 'RightHandThumb3_Xrotation'
_mapping['ip_flexion'] = 'RightHandThumb4_Xrotation'

# the rest of the fingers, we can automate
for f in [2, 3, 4, 5]:
    _mapping['mcp%d_flexion'%f] = _map_to_leap('mcp%d_flexion'%f)
    _mapping['mcp%d_abduction'%f] = _map_to_leap('mcp%d_abduction'%f)
    _mapping['pm%d_flexion'%f] = _map_to_leap('pm%d_flexion'%f)
    _mapping['md%d_flexion'%f] = _map_to_leap('md%d_flexion'%f)

joint_map = _mapping # use OrderedDict for reliability
joint_names = list(joint_map)

def load_calibration_run(layout, sub, run):
    '''
    loads one run's data in sklearn-compatible format

    Parameters
    -----------
    layout : bids.BIDSLayout
    sub : str
    run : int

    Returns
    -------
    glove_data : an (n_samples, n_sensors) np.array
        the glove measurements from this run
    leap_data : (n_samples, n_joints)
        the joint angle measurements from the leap motion

    Notes
    ----------
    The glove data is interpolated to the timestamps of the
    leap motion data, so `n_samples` refers to the number of
    leap motion samples (cropped to the time recordings overlap) .
    '''
    glove_f, joint_f = layout.get(
        subject = sub,
        task = 'calibration',
        run = run,
        suffix = 'motion',
        extension = 'tsv'
    )
    joint_data = joint_f.get_df()
    glove_data = glove_f.get_df()

    # crop joint angles
    tmin, tmax = glove_data.LATENCY.min(), glove_data.LATENCY.max()
    joint_data = joint_data.set_index('LATENCY')[tmin:tmax]

    # interpolate glove measurements to joint angle timestamps
    gd = glove_data.loc[:, glove_data.columns != 'LATENCY'].to_numpy()
    interp_func = interp1d(x = glove_data.LATENCY, y = gd, axis = 0)
    gd_interp = interp_func(joint_data.index)

    # return X, y (just the columns we need), and run
    y = joint_data[[val for key, val in joint_map.items()]]
    run_col = np.full(joint_data.shape[0], run)
    return gd_interp, y.to_numpy(), run_col

def load_calibration_runs(layout, sub):
    '''
    loads all calibration runs from one subject
    into an sklearn-friendly format

    Parameters
    -----------
    layout : bids.BIDSLayout
    sub : str

    Returns
    -------
    glove_data : an (n_samples, n_sensors) np.array
        the glove measurements from all calibration runs
    leap_data : an (n_samples, n_joints) np.array
        corresponding joint angle measurements from the leap motion
    run : an (n_samples,) np.array
        the index of the run each sample came from
    '''
    Xs = []
    ys = []
    runs = []
    run_idxs = layout.get_runs(subject = sub, task = 'calibration')
    for run in run_idxs:
        X, y, r = load_calibration_run(layout, sub, run)
        Xs.append(X)
        ys.append(y)
        runs.append(r)
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    run = np.concatenate(runs)
    return X, y, run

def compute_calibration(layout, sub):
    '''
    Estimates the linear mapping from glove sensors to joint angles
    from parallel recordings during calibration runs recorded outside the MRI.

    Between-run cross-validation scores and the fitted linear mapping are saved
    in a BIDS derivatives directory under the `calibration` workflow.

    Parameters
    ----------
    layout : bids.BIDSlayout
    sub : str

    Returns
    ----------
    mapping : sklearn.pipeline.Pipeline
        the linear mapping fit to all calibration data
    '''

    ## load data
    glove, joints, runs = load_calibration_runs(layout, sub)

    ## evaluate leave-one-run-out cross-validation score of linear mapping
    pipe = make_pipeline(
        StandardScaler(),
        Ridge(alpha = 1.)
    )
    scores = np.stack([
        cross_val_score(
            pipe,
            glove, joints[:, jnt], groups = runs,
            cv = LeaveOneGroupOut(),
            scoring = 'r2'
        ) for jnt in range(joints.shape[1])
    ])
    res_dict = {jnt: scr for jnt, scr in zip(joint_map, scores.mean(1))}
    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'calibration'
    )
    fpath = sink.get_path(
        subject = sub,
        task = 'calibration',
        desc = 'mapping',
        datatype = 'motion',
        suffix = 'r2',
        extension = '.json'
    )
    with open(fpath, 'w') as f:
        json.dump(res_dict, f, indent = 4)

    ## and save linear mapping fitted to all calibration data
    pipe.fit(glove, joints)
    fpath = fpath.replace('_r2.json', '_reg.joblib')
    dump(pipe, fpath) # save model
    return pipe

def get_calibration(layout, sub):
    '''
    loads the glove-->joint mapping for one subject

    Parameters
    ----------
    layout : bids.BIDSLayout
    sub : str

    Returns
    ----------
    mapping : sklearn.pipeline.Pipeline
        the linear mapping fit to all calibration data
    '''
    # index calibration derivatives if haven't already
    calib_dir = os.path.join(layout.root, 'derivatives', 'calibration')
    if os.path.exists(calib_dir) and (not layout.derivatives):
        warn('Adding %s to BIDSLayout...'%calib_dir)
        layout.add_derivatives(calib_dir)
    # try reading mapping from calibration file
    matches = layout.get(subject = sub, suffix = 'reg')
    if matches:
        return load(matches[0].path)
    # if we haven't found a mapping, then we just need to make one
    print("Didn't find pre-computed glove calibration for sub-%s..."%sub)
    print('Computing one now from calibration data.')
    mapping = compute_calibration(layout, sub)
    layout.add_derivatives(calib_dir)
    return mapping

def load_joint_angles(layout, sub, run, myo_fix):
    '''
    loads joint angles for a motor task run

    Parameters
    ----------
    layout : bids.BIDSLayout
    sub : str
    run : int
        Run of motion tracking task, a.k.a. `motor` in BIDS directory
    myo_fix : bool
        The neural network we use in this project was trained with a version
        of myosuite's myohand model that mixes up flexion and adduction for
        the CMC joint and inverts the sign of all thumb joints, see issue:
        https://github.com/MyoHub/myosuite/issues/112. If `myo_fix` is true,
        the `joint_angles` dataframe output by this function does the same
        to ensure compatibility.

    Returns
    ---------
    joint_angles : an (n_samples, n_angles) pd.DataFrame
        Joint angles, with column names in myosuite nomenclature
        and units converted to radians.
    timestamps : an (n_samples,) np.array
    sfreq : int
        The nominal sampling rate of the glove data.

    Notes
    --------
    The joint angle names in the R2 score files in the calibration derivatives
    directory are always as if `myo_fix = False`, since it seemed confusing
    to have deliberately misnamed values in saved files.
    '''
    # load subject's glove calibration
    mapping = get_calibration(layout, sub)
    # load glove data for given task run
    sidecar_f, glove_f = layout.get(
        subject = sub,
        task = 'motor',
        run = run,
        suffix = 'motion'
    )
    df = glove_f.get_df()
    timestamps = df.LATENCY.to_numpy()
    glove_sensors = df.loc[:, df.columns != 'LATENCY'].to_numpy()

    # and predict joint angles
    joint_angles = mapping.predict(glove_sensors)
    # then reformat for myosuite before returning
    df = pd.DataFrame(-1.*joint_angles, columns = joint_names)
    if myo_fix: # flip sign of thumb angles
        df.loc[:, df.columns.str.contains('cmc|mp|ip')] *= -1
        df = df.rename(columns = {
            'cmc_abduction': 'cmc_flexion',
            'cmc_flexion': 'cmc_abduction'
        })
    df = df.apply(np.deg2rad)
    return df, timestamps, sidecar_f.get_dict()['SamplingFrequency']
