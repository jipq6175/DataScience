# regression julia
cd("G:\\My Drive\\18. Github Repo\\DataScience");

# load libraries
using Plots, Statistics, StatsBase, DataFrames, XLSX, MLBase, RDatasets, GLMNet, DecisionTree, LIBSVM;

# Load the first data to play with
R = XLSX.readxlsx(".\\data\\zillow_data_download_april2020.xlsx");
sale_counts = R["Sale_counts_city"][:];
df_sale_counts = DataFrame(sale_counts[2:end, :], Symbol.(sale_counts[1, :]));
monthly_listings = R["MonthlyListings_City"][:];
df_monthly_listings = DataFrame(monthly_listings[2:end,:], Symbol.(monthly_listings[1,:]));

# get only the 2020-02 data
monthly_listings_2020_02 = df_monthly_listings[!,[1,2,3,4,5,end]];
rename!(monthly_listings_2020_02, Symbol("2020-02") .=> Symbol("listings"));
sale_counts_2020_02 = df_sale_counts[!,[1,end]];
rename!(sale_counts_2020_02, Symbol("2020-02") .=> Symbol("sales"));

# join two dataframes
feb2020 = innerjoin(monthly_listings_2020_02, sale_counts_2020_02, on=:RegionID);
dropmissing!(feb2020);


# different data
cats = dataset("MASS", "cats");

lmap = labelmap(cats[!,:Sex]);
ci = labelencode(lmap, cats[!,:Sex]);
scatter(cats[!,:BWt],cats[!,:HWt],color=ci,legend=false)
