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