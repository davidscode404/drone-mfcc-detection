# drone-mfcc-detection

## Results

## Setup

Be sure to use a virtual environment:
> ```sh
> $ python3 -m venv venv
> ```
(Here, `venv` is just an example, the virtual environment can be given *any* name).

Activate the virtual environment:
> ```sh
> $ source venv/bin/activate
> ```

Upon activation, run `pip list`. Only `pip` and `setuptools` should be installed in the virtual environment.

Install requirements:
> ```sh
> $ pip install -r requirements.txt
> ```

## Run

Run `classification.py`:
> ```sh
> $ python3 classification.py
> ```

Six `png`s will be saved to results:

Note that each `png` is prefaced with a timestamp (e.g. `20260504_143022_`).