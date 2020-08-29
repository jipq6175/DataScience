# change dir
cd("G:\\My Drive\\18. Github Repo\\DataScience");

# versioninfo
versioninfo()

# load some analysis and data handling libs
using BenchmarkTools, DataFrames, DelimitedFiles, CSV, XLSX


# get the data
# download it as .csv file
P = download("https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv", "programminglanguages.csv");

# read in the csv file
# P is Matrix{Any} very hard to work with
P, H = readdlm("programminglanguages.csv", ','; header=true);

# using CSV into DataFrame
C = CSV.read("programminglanguages.csv", DataFrame);

# look at and access data frame
C[1:10, :]
C.year
names(C)

# short summary
describe(C)

# test the performances
@btime P, H = readdlm("programminglanguages.csv", '.', header=true);
@btime C = CSV.read("programminglanguages.csv", DataFrame);

# wwrite csv
CSV.write("test.csv", C);


# Switch to XLSX
T = XLSX.readdata(".\\data\\zillow_data_download_april2020.xlsx", "Sale_counts_city", "A1:F9");
typeof(T)

# Matrix Any is hard to work with
df = DataFrame(T[2:end, :]);
rename!(df, names(df) .=> Symbol.(T[1,:]))
# equiv to
df = DataFrame(T[2:end, :], Symbol.(T[1, :]))

# convert everything the reasonable data type
nm = names(df);
for name in nm
    type = typeof(df[2, Symbol(name)]);
    try
        df[:, Symbol(name)] = convert(Array{type, 1}, df[:, Symbol(name)]);
    catch er;
    end
end
df

# read the whole table
@time G = XLSX.readtable("data/zillow_data_download_april2020.xlsx","Sale_counts_city");
dg = DataFrame(G...)

# convert the dg into proper type
nm = names(dg);
for name in nm
    type = typeof(dg[2, Symbol(name)]);
    try
        dg[!, Symbol(name)] = convert(Array{Union{Missing, type}, 1}, dg[!, Symbol(name)]);
    catch er;
        println("Column $name missing value detected ... ");
        throw(er);
    end
end
dg

# using data frame:C to find the language
function find_lang(data::DataFrame, lang::String)
    idx = findfirst(data[!, :language] .== lang);
    if !isnothing(idx)
        return data[idx, :year];
    else
        @warn("-- $lang not found... ");
    end
end

find_lang(C, "Julia")


# using the dataframe to find number of languages
function number_lang(data::DataFrame, year::Int64)
    ans = length(findall(data[!, :year] .== year));
    return ans;
end

number_lang(C, 2011)
