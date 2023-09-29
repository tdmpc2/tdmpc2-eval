# TD-MPC2 Evaluation

*Anonymized in compliance with ICLR 2024 submission guidelines.*<br/><br/>

Evaluation repository for the [TD-MPC2](https://www.tdmpc2.com) project by

[Anonymous Authors](https://www.tdmpc2.com)<br/><br/>

[[Website]](https://www.tdmpc2.com) [[Paper]](https://openreview.net/pdf?id=Oxh5CstDJU)  [[OpenReview]](https://openreview.net/forum?id=Oxh5CstDJU) [[Models]](https://www.tdmpc2.com/models)  [[Dataset]](https://www.tdmpc2.com/dataset)

<br/><br/>
Code for training TD-MPC2 agents will be released at a later date.

----

## Getting started

You will need a machine with a GPU and at least 12 GB of RAM. A GPU with at least 8 GB of memory is required for evaluation of the 317M parameter models.

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
cd docker && docker build . -t <user>/tdmpc2:0.1.0
```

If you prefer to install dependencies manually, start by installing dependencies via `conda` by running

```
conda env create -f docker/environment.yml
```

If you want to run ManiSkill2, you will additionally need to download and link the necessary assets by running

```
python -m mani_skill2.utils.download_asset all
```

which downloads assets to `./data`. You may move these assets to any location. Then, add the following line to your `~/.bashrc`:

```
export MS2_ASSET_DIR=<path>/<to>/<data>
```

and restart your terminal. Meta-World additionally requires MuJoCo 2.1.0. We host the unrestricted MuJoCo 2.1.0 license (courtesy of Google DeepMind) at [https://www.tdmpc2.com/files/mjkey.txt](https://www.tdmpc2.com/files/mjkey.txt). You can download the license by running

```
wget https://www.tdmpc2.com/files/mjkey.txt -O ~/.mujoco/mjkey.txt
```

See `docker/Dockerfile` for installation instructions if you do not already have MuJoCo 2.1.0 installed. MyoSuite requires `gym==0.13.0` which is incompatible with Meta-World and ManiSkill2. Install separately with `pip install myosuite` if desired. Depending on your existing system packages, you may need to install other dependencies. See `docker/Dockerfile` for a list of recommended system packages.

----

## Supported tasks

We currently support **104** continuous control tasks from **DMControl**, **Meta-World**, **ManiSkill2**, and **MyoSuite**. See below table for expected name formatting for each task domain:

| domain | task
| --- | --- |
| dmcontrol | dog-run
| dmcontrol | cheetah-run-backwards
| metaworld | mw-assembly
| metaworld | mw-pick-place-wall
| maniskill | pick-cube
| maniskill | pick-ycb
| myosuite  | myo-hand-key-turn
| myosuite  | myo-hand-key-turn-hard

which can be run by specifying the `task` argument for `evaluation.py`. Multi-task evaluation is specified by setting `task=mt80` or `task=mt30` for the 80 and 30 task multi-task checkpoints, respectively.


## Example usage

See below examples on how to evaluate downloaded single-task and multi-task checkpoints.

```
$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
```

All single-task checkpoints expect `model_size=5`. Multi-task checkpoints are available in multiple model sizes. Available arguments are `model_size={1, 5, 19, 48, 317}`. Note that single-task evaluation of multi-task checkpoints is currently not supported. See `config.yaml` for a full list of arguments.

----

## Citation

If you find our work useful, please consider citing the paper as follows:

```
@article{Anonymous2023TDMPC2,
	title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
	author={Anonymous Authors},
	booktitle={Twelfth International Conference on Learning Representations (Submission)},
	url={https://openreview.net/forum?id=Oxh5CstDJU},
	year={2023}
}
```

----

## Contributing

You are very welcome to contribute to this project. However, please understand that we are unable to respond to any issues or pull requests for the duration of the double-blind review period.

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
