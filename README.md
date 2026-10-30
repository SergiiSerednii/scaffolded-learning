# scaffolded-learning

PyTorch implementation of **scaffolded learning** concept: training a model on a main target together with auxiliary (scaffold) targets predicted at intermediate stages.

The package provides:

- **Building blocks** (`scaffolded_learning.blocks`): `BackboneModule`, `ScaffoldModule` and `HeadModule`.
- **Predefined models** for popular architectures:

| Model | Data | Scaffold | Output |
| --- | --- | --- | --- |
| `ScaffoldedMLPBinary` | tabular | one binary head per scaffold column | binary |
| `ScaffoldedMLPMultiClass` | tabular | one multiclass head | multiclass |
| `ScaffoldedVGG` | images, VGG16-BN | one multiclass head | multiclass |
| `ScaffoldedGPT` | token ids | one multiclass head | multiclass |

```bash
pip install scaffolded-learning
```

Python ≥ 3.8. Depends on `torch`, `numpy`, `pandas`, `scikit-learn`, `pyarrow`, `optuna`.