# Data-Wrangling-Final-Project

## What Really Drives Diamond Prices?

### Overview

For this project, I looked at what actually determines diamond prices. Most people assume that cut, color, and clarity are the biggest contributors, but once I started working with the dataset, it became clear that carat size has the strongest influence on price. I cleaned the data, made several visualizations, and ran a regression model to see how each characteristic affects price.

### Why This Topic Matters

Understanding how diamonds are priced helps show how markets assign value and how consumers often misunderstand what really drives cost. A lot of people think that a flawless diamond or a perfect color rating automatically means a huge price jump, but the data doesn't really support that idea. This makes the topic interesting from both an economic and a behavioral standpoint.

### Key Findings

#### Carat Is the Most Important Factor

Carat size ended up being the clearest and strongest predictor. Prices rise almost exponentially as carat increases, and even a small bump in carat size tends to create a big jump in price. When I graphed the data, the upward trend was obvious.

#### Cut Matters, But Not Nearly as Much

Cut quality does influence price, but the changes aren't dramatic when compared to carat. The cut mainly affects how the diamond looks rather than how much the market values it. Even the highest cut grades only showed modest differences in average price.

#### Color and Clarity Have a Weak Impact

Color and clarity didn't change the price as much as I expected. The price ranges for each grade overlapped a lot, and the median prices stayed relatively close across the different categories. These qualities just don't separate diamonds very much in terms of value, at least compared to carat.

#### Regression Results Back This Up

The OLS regression confirmed everything the visuals suggested. Carat explained the majority of the variation in price, and the coefficient was extremely large. Carat was also highly significant statistically, while cut, color, and clarity barely made a noticeable impact once carat was included in the model.

### Methods

To complete the project, I cleaned and organized the dataset, removed outliers where necessary, created new variables such as binned carat categories, and built multiple visualizations to compare diamond characteristics. I finished by running an OLS regression model to measure how strong each predictor was relative to the others. I used common data-wrangling and plotting tools in Python or R, depending on what the assignment required.

### What I Learned

This project showed me that carat size drives diamond pricing far more than most people realize. The qualities consumers usually pay attention to, like color and clarity, don't affect the price nearly as much. Working through the analysis also helped me understand how visualizations and regression models can reveal patterns that challenge common assumptions. 
