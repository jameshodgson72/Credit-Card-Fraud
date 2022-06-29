"""This is my EDA class.  The aim is to capture some useful functions/tools which
I can re-use throughout different projects.  The functions should save me a bit of
time on basic EDA activities.  I will add to them over time.  Initially they will
be basic and lack some validation and exception handling which will be addressed over
time."""
# import to use for checking if columns are of string type, i.e. non-numeric
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class Eda:
    def __init__(self, DataFrame):
        """Function to instantiate the class - takes a single argument of the data frame
        which is to be analysed.  Then any subsequent calls to class methods will use that
        data frame as the data frame to enact the methods on."""
        try:
            self.df = DataFrame
        except ValueError:
            print("EDA class init function requires a data frame argument")
            
    def cols_with_nulls(self):
        """Function to return a list of columns in our dataframe containing nulls"""
        return self.df.columns[self.df.isnull().any()].tolist()
    
    def nulls_for_col(self, col_name):
        """Function to return the number of nulls in a given column"""
        return self.df[col_name].isnull().value_counts().loc[True]
    
    def nulls_per_col(self, col_list):
        """Function to return the number of nulls for each column in a supplied column list."""
        for col in col_list:
            print( col, " : ", self.nulls_for_col(col))
        return
    
    def report_nulls(self):
        """function to report on the number of nulls in a data frame, also reporting
        the proportion of rows which have one or more nulls in them.  Also lists how
        many nulls appear in each column in descending order."""
        print( "Total nulls in data frame is: ", self.df.isnull().sum().sum() )
        print( (self.df.shape[0] - self.df.dropna().shape[0]), " out of ", self.df.shape[0], 
              "rows have nulls in one or more column" )
        print( self.df[self.df.columns[self.df.isnull().any()]].isnull().sum().sort_values(ascending=False) )
        return

    def report_string_cols(self):
        """function to report on 'string' columns so you know how many columns you have
        containing strings and what their unique values are."""
        num_string_cols = 0
        for col in self.df.columns.sort_values():
            if is_string_dtype( self.df[col].dtype ):
                print( col, ":" )
                print( self.df[col].unique() )
                print( "---------------------\n\n" )
                num_string_cols += 1
        print( num_string_cols, " string columns in total")
        return
    
    def report_col(self, col_name):
        """function to provide some information on a column - how many unique values it contains
        and what these are."""
        values = self.df[col_name].unique()
        print( "Column: '", col_name, "' with ", len(values), " unique values:" )
        print( values )
        print( "---------------------\n\n" )
        return
        
    def report_all_cols(self):
        """function to report on all the columns in the dataframe and their unique values."""
        for col in self.df.columns.sort_values():
            self.report_col(col)
        return
    
    def return_outlier_ids_for_col(self, col_name, devs=3):
        """ function to return a set of row Ids for the rows which contain outliers in a given column.
        This is worked out by N standard deviations from the mean.  By default, N=3, but you can specify
        another value with the devs argument."""
        mean = self.df[col_name].mean()
        std_dev = self.df[col_name].std()
        upper_bound = mean + (devs * std_dev)
        lower_bound = mean - (devs * std_dev)
        return self.df[(self.df[col_name] > upper_bound) | (self.df[col_name] < lower_bound)].index.tolist()
    
    def report_outliers_for_col(self, col_name, devs=3):
        """Function to list out the outlier values in a given column.  By default these are 3 the
        values which are 3 standard deviations from the mean but you can change that with the devs
        parameter."""
        print( "Outliers for column: ", col_name )
        print( self.df.loc[self.return_outlier_ids_for_col(col_name, devs)][col_name] )
        print( "----------------------\n\n")
        return
    
    def report_all_outliers(self, devs=3):
        """Function to report the number of outliers in each column in a data frame.  Only
        columns containing numeric values will be reported on."""
        for col in self.df.columns.sort_values():
            if is_numeric_dtype(self.df[col].dtype):
                num_outliers = len(self.return_outlier_ids_for_col(col, devs))
                print("Column: ",col," has ", num_outliers, " outliers.")
        return;
    
    def delete_outliers_for_col(self, col_name, devs=3):
        """Function to delete outliers from a given column.  By default this will be data points
        more than 3 standard deviations from the mean, but you can specify a different number of
        standard deviations to use via the 'devs' argument"""
        self.df.drop(labels=self.return_outlier_ids_for_col(col_name, devs), inplace=True, axis=0)
        return

