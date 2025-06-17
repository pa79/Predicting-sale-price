#  Product Price Prediction from Marketplace Listings

This project focuses on building a machine learning model to predict the sale price of online product listings using structured and unstructured data provided by sellers. Accurately estimating product prices is a challenging task influenced by subtle differences in product descriptions, conditions, and categories—particularly in competitive online marketplaces.

##  Objective

The goal is to predict the final sale price of items based on features such as item condition, category, brand name, shipping information, and text descriptions. This task simulates a real-world application faced by e-commerce platforms when setting or suggesting optimal prices.

##  Dataset

The dataset consists of two tab-delimited files:

- `train.tsv`: Contains ~700,000 labeled listings with sale prices.
- `test.tsv`: Contains ~3.5 million listings without prices (to be predicted).

Each listing includes:
- `name`: Product title (with price mentions removed and replaced by `[rm]`).
- `item_condition_id`: Condition of the item.
- `category_name`: Hierarchical category of the product.
- `brand_name`: Brand of the item.
- `price`: **Target variable** (only in `train.tsv`).
- `shipping`: Indicates who pays for shipping (0 = buyer, 1 = seller).
- `item_description`: Detailed product description (also cleaned of prices).

##  Modeling

The predictive model is trained on the `train.tsv` data and evaluated on `test.tsv`. To account for computational complexity, model development begins with a representative sample of the training data.

Model performance is evaluated using **Root Mean Squared Logarithmic Error (RMSLE)** to penalize underestimation more than overestimation and handle large-scale price ranges effectively.

##  Evaluation Metric

**RMSLE Formula:**

`ε = sqrt( (1/N) * Σ[ log(p̂ᵢ + 1) - log(pᵢ + 1) ]² )`

Where:
- p̂ᵢ is the predicted price  
- pᵢ is the actual price  
- N is the number of predictions

