# Multi-Region Transfer Learning for Segmentation of Crop Field Boundaries in Satellite Images with Limited Labels

Authors: Hannah Rae Kerner, Saketh Sundar, Manthan Satish

This repository contains the code implementation for the paper "Multi-Region Transfer Learning for Segmentation of Crop Field Boundaries in Satellite Images with Limited Labels" accepted at the 2023 AAAI Workshop on AI to Accelerate Science and Engineering (AI2ASE) 2023.

## Usage

### Environment Setup

To create a new environment, utilize the provided YAML file:

```bash
conda env create -f environment.yml
```

### Downloading Data

Use the links below to download the data for each region. Some contain only the labels while others contain the images as well
- [Labels for Rwanda](https://beta.source.coop/nasa/rwanda-field-boundary-competition/)
- [Labels for Kenya](https://beta.source.coop/radiantearth/african-crops-kenya-01/)
- [Labels for France](https://www.data.gouv.fr/fr/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/)
- [Data for France](https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/field_delineation.html)
<!-- - [Data for South Africa]() -->

Place the downloaded data in the `data` folder.

#### Download Images from Google Earth Engine (GEE):

Utilize the `.js` script located in `data_helpers/gee_images_downloader.js` to download images from Google Earth Engine.

### Processing Data

The data should be stored in the `data` folder. It will be stored in the following structure:
- **images_mar**, **images_jun**, **images_sep**: Directories containing satellite images corresponding to different months (March, June, September).
- **masks**: Directory containing labeled masks or ground truth data for crop field boundaries.
- **masks_filled**: Directory potentially filled with processed or augmented mask data if applicable.

It will look something like this:

```
data
├── country
│   ├── images_mar
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── ...
│   ├── images_jun
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── ...
│   ├── images_sep
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── ...
│   ├── masks
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── ...
│   ├── masks_filled
│   │   ├── image_1.png
│   │   ├── image_2.png
│   │   ├── ...
├── ...
```

### Training

To train the model, run the following command:

```bash
python train.py --config config.yaml
```

### Fine Tuning

To fine tune the model, run the following command:

```bash
python fine_tune.py --config config.yaml
```

### Testing

To test the model, run the following command:

```bash
python test.py --config config.yaml
```

### Inference

To run inference on the model, run the following command:

```bash
python inference.py --config config.yaml
```

## Citation

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{hkerner2023multitlf,
    title={Multi-Region Transfer Learning for Segmentation of Crop Field Boundaries in Satellite Images with Limited Labels},
    author={Satish, Manthan and Kerner, Hannah Rae and Sundar, Saketh},
    journal={AAAI Workshop on AI to Accelerate Science and Engineering (AI2ASE)},
    year={2023}
}
```