import pandas as pd
import logging

from gspread.utils import rowcol_to_a1

class DataValue():
    """Class for individual data points in a database
    """
    def __init__(self, datamart: str, sheet: str, index, column: str, value, df: pd.DataFrame = None, index_col = None, gindex: int = None, gcolumn: int = None):
        """Init for DataValue object
        
        Arguments:
        self
        datamart (str): name of the table/datamart
        sheet (str): name of the sheet
        index (str or int) : name of the index row (primary_key)
        column (str) : name of the column
        value: cell value
        index_col: index (primary key) column (used only for row detection). If None, keeps the composiiton. Default is None.
        df: related DataFrame
        gindex (int): row index as an int (starting with 1)
        gcolumn (int): column index as an int (staring with 1)
        include_index (bool): whether to include df index, default is True
        """
        self.datamart = datamart
        self.sheet = sheet
        self.index = index
        self.column = column
        if gindex is None:
            if df is None: raise TypeError("Must specify df if gindex is None")
            if index_col is None:
                self.gindex = df.index.get_loc(index) + 1  # Plus 1 as google indexing starts with 1, not 0
            else:
                self.gindex = df.reset_index().set_index(index_col).index.get_loc(index) + 1  # Plus 1 as google indexing starts with 1, not 0
            logging.debug(f"gindex for {index} is {gindex}")
        else:
            if gindex is None: raise TypeError("Must specify gindex if df is None")
            if gindex < 1: raise ValueError("gindex cannot be below 1, as starting row in Google Sheets has index 1")
            self.gindex = gindex 
        if gcolumn is None:
            if df is None: raise TypeError("Must specify df if gcolumn is None")
            self.gcolumn = df.columns.get_loc(column) + 1  # Plus 1 as google indexing starts with 1, not 0
            logging.debug(f"gcolumn for {column} is {gcolumn}")
        else:
            if gcolumn is None: raise TypeError("Must specify gindex if df is None")
            if gcolumn < 1: raise ValueError("gcolumn cannot be below 1, as starting column in Google Sheets has index 1")
            self.gcolumn = gcolumn
        self.value = value
        logging.debug(f"Initialised {self}")

    # aliases
    @property
    def table(self):
        return self.datamart

    @property
    def row(self):
        return self.index

    @property
    def a1(self):
        """returns cell address in A1 notation"""
        return rowcol_to_a1(self.gindex, self.gcolumn)

    @property
    def A1(self):
        """alias for self.a1""" 
        return self.a1

    def __repr__(self):
        return f"DataValue: datamart={self.datamart}, sheet={self.sheet}, row={self.index} ({self.gindex}), column={self.column} ({self.gcolumn}), value={self.value}"

def get_datavalue_dict(datavalues):
    """Returns a dict from a list of datavalues
    
    Dict example:
    {'datamart1': {'datasheet1': [DataValue: datamart=datamart1, sheet=datasheet1, row=index1 (1), column=column1 (1), value=value1]},'datamart2': {'datasheet2': [DataValue: datamart=datamart2, sheet=datasheet2, row=index1 (1), column=column2 (2), value=value2]}}
     """
    data_dict = dict()
    for datavalue in datavalues:
        if not isinstance(datavalue, DataValue):
            raise TypeError("items in datavalues list must be of class DataValue")
        if not datavalue.datamart in data_dict:
            data_dict[datavalue.datamart] = dict()
        if not datavalue.sheet in data_dict[datavalue.datamart]:
            data_dict[datavalue.datamart][datavalue.sheet] = list()
        data_dict[datavalue.datamart][datavalue.sheet] += datavalue,
    return data_dict
        
def test():
    print(f"TEST INITIALISED: {__file__}")
    logging.info(f'{__file__} test initialised')
    dv1 = DataValue('int_main_copy', 'i_main', '35', 'c_name_en', 'WORKS!', gindex=35, gcolumn=6)
    dv2 = DataValue('int_main_copy', 'i_main', '36', 'c_name_en', 'WORKS!11111', gindex=36, gcolumn=6)
    testdict = get_datavalue_dict([dv1, dv2])
    print(testdict)
    return [dv1, dv2]

if __name__ == '__main__':
    test()