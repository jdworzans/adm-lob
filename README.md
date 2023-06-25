# LOB (Limit Order Book) Data Mining

# Data
[Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods](https://arxiv.org/abs/1705.03233)
introduced benchmark dataset. The dataset can be downloaded from
[this link](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649).

Details of dataset are described in an article.
For simplicity, I will focus on
the version of dataset limited
to events between 10:30 and 18:00,
since authors suggest events outside this period
has not comparable dynamic.
Such limited dataset is available
in `BenchmarkDatasets/NoAuction` subdirectory
(in contrast to `BenchmarkDatasets/Auction` subdirectory, containing complete data).


Another available source of data is [LOBSTER](https://lobsterdata.com/).

# Problem
Problem is to predict wheter in short horizon stock price goes up, down or stays approximately same (3-class classification),
base on LOB (Limit Order Book) with depth 10 (10 highest bid prices and volumes and lowest ask prices and volumes)

# Methods
Across my project I focused on reproducing [DeepLOB](https://arxiv.org/abs/1808.03668) results.
I've implemented training and evaluation process in `lightning` framework,
based on original [implementation](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/tree/master) of authors.

# Key issues
Preparation and preprocessing were optimized due
to very large original RAM usage (more than 32GiB).
Finally, preprocessing and loading data last the same,
but can be done on almost every CPU (~2GiB of RAM),
only by avoiding repeating memory allocation for
already existing arrays.

It occurs, that according to the authors' code,
it wasn't tested in same exact fashion,
as suggested by benchmark authors.
Metrics obtained are lower, and their standard deviation
(reported for baseline benchmark methods)
is very large.
