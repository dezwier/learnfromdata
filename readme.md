# Learnfromdata

## Contents

~~~~
  learnfromdata             # Learnfromdata library       
  ├── lfd                        # Learnfromdata source code
  │   ├── config                    # Configuration tools
  │   ├── data                      # Data class
  │   ├── transform                 # Transform classes
  │   ├── model                     # Model classes
  │   ├── pipeline                  # Pipeline class
  │   ├── utils                     # Useful functions
  │   └── visualize                 # Visualizer class
  ├── notebooks                 # Examplary notebooks using LFD
  ├── tests                     # Unittests for source code
  └── setup.py                  # Configuration file for package build
~~~~

## Commands

Dependencies

* `conda create --name <env_name> python=3.9.18`
* `conda activate <env_name>`
* `pip install -r requirements.txt` in a python 3.7 environment

## UML of LFD module
<pre style="line-height: 1.1; letter-spacing: -0.2px; font-size: small;">
┌─────────────┐           ┌─────────────────┐           ┌───────────────┐       ┌──────────────┐
│ Pipeline    ├───────────┤ Bootstrap       │           │ Plotter       ├───────┤ PlotterModel │
├─────────────┤           ├─────────────────┤           ├───────────────┤       ├──────────────┤
│ learn       │           │ learn_pipelines │           │ line_chart    │       │ run_app      │
│ apply       │           │ get_meta        │           │ boxplots      │       │ liftcurve    │
│ save        │           └─────────────────┘           │ histogram     │       │ confusion    │
│ load        ├─────┐                                   │ ...           │       │ ...          │
└─────┬───────┘     │                                   └───────────────┘       └──────────────┘
      │             │     ┌─────────────┐
      │             ├─────┤ Transformer ├──────────┬───────────────┬────────────┬─────────────┬───────────────┬──────────────┬─────────────────┐
┌─────┴───────┐     │     ├─────────────┤          │               │            │             │               │              │                 │
│ Data        │     │     │ learn       │   ┌──────┴──────┐   ┌────┴───┐   ┌────┴────┐   ┌────┴────┐   ┌──────┴─────┐   ┌────┴─────┐   ┌───────┴──────┐
├─────────────┤     │     │ apply       │   │ UniSelector │   │ Binner │   │ Imputer │   │ Encoder │   │ Biselector │   │ Expander │   │ Standardizer │
│ merge       │     │     │ save        │   ├─────────────┤   ├────────┤   ├─────────┤   ├─────────┤   ├────────────┤   ├──────────┤   ├──────────────┤
│ split       │     │     │ load        │   │ learn       │   │ learn  │   │ learn   │   │ learn   │   │ learn      │   │ learn    │   │ learn        │
│ balance     │     │     └─────────────┘   │ apply       │   │ apply  │   │ apply   │   │ apply   │   │ apply      │   │ apply    │   │ apply        │
│ filter      │     │                       └─────────────┘   └────────┘   └─────────┘   └─────────┘   └────────────┘   └──────────┘   └──────────────┘
│ select      │     │     ┌─────────────┐
│ sample      │     └─────┤ Model       ├──────────┬──────────────┬──────────────┬─────────────────┬────────────────┬───────────────┐
│ concat      │           ├─────────────┤          │              │              │                 │                │               │
│ generate    │           │ learn       │   ┌──────┴─────┐   ┌────┴────┐   ┌─────┴─────┐   ┌───────┴──────┐   ┌─────┴────┐   ┌──────┴──────┐
│ add_noise   │           │ apply       │   │ Regression │   │ Xgboost │   │ NeuralNet │   │ DecisionTree │   │ Isotonic │   │ UniSelector │
│ save        │           │ evaluate    │   ├────────────┤   ├─────────┤   ├───────────┤   ├──────────────┤   ├──────────┤   ├─────────────┤
│ load        │           │ explain     │   │ learn      │   │ learn   │   │ learn     │   │ learn        │   │ learn    │   │ learn       │
└─────────────┘           │ save        │   │ apply      │   │ apply   │   │ apply     │   │ apply        │   │ apply    │   │ apply       │
                          │ load        │   └────────────┘   └─────────┘   └───────────┘   └──────────────┘   └──────────┘   └─────────────┘
                          └─────────────┘
</pre>