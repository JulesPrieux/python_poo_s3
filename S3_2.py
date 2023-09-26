from datetime import datetime
from uuid import uuid4

import pandas as pd
import yfinance as yf
import numpy as np


class Quote:
    def __init__(self, date: datetime, price: float):
        self.date = date
        self.price = price


class FinancialAsset:
    def __init__(self, ticker, quote, currency):
        self.ticker: str = ticker
        self.last_quote: Quote = quote
        self.currency: str = currency
        self.history: [Quote] = []

    def update_price(self, new_quote: Quote):
        self.history.append(self.last_quote)
        self.last_quote = new_quote

    def populate_quote_history_from_df(self, df_data: pd.DataFrame):
        """
            for index, row in df.iterrows():
            quote = Quote(date=index.to_pydatetime(), price=row['Close'])
            self.quotes.append(quote)
        """
        dates = df_data.index.to_pydatetime().tolist()
        prices = df_data['Close'].tolist()
        self.quotes = [Quote(date, price) for date, price in zip(dates, prices)]

    def quotes_to_dataframe(self) -> pd.DataFrame:
        data = {
            "Date": [quote.date for quote in self.quotes],
            "Price": [quote.price for quote in self.quotes]
        }
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df

    def get_description(self):
        print(f'The ticker for this asset is {self.ticker} and its price is {self.price} {self.currency}')


class Equity(FinancialAsset):
    def __init__(self, ticker, last_quote, currency, dividend):
        super().__init__(ticker, last_quote, currency)
        # alternative way to do it : FinancialAsset.__init__(self, ticker, price, currency)
        self.dividend: float = dividend

    def get_description(self):
        print(f'The ticker for this asset is {self.ticker} and its price is {self.last_quote.price} {self.currency}.'
              f' Last dividend was {self.dividend}')

    def calculate_pe_ratio(self):
        if self.dividend == 0:
            return float('inf')  # P/E is theoretically infinite if earnings are zero
        return self.last_quote.price / self.dividend


"""
Exercise 1 : Yahoo Finance Data Loader

In this exercise, you will design a class, `YahooFinanceDataLoader`, that will act as a data loading utility. 
Yahoo Finance provides stock market data, and this utility will help you fetch stock data in a structured manner 
using the `yfinance` library.

**Pre-requisites**:
1. Make sure you have `yfinance` library installed:
    If not run one of the following command based on your interpreter
   ```
   pip install yfinance
   conda install -c conda-forge yfinance
   ```

**Objective**:
You have been provided with a blueprint of the `YahooFinanceDataLoader` class. You need to implement the following:

1. `_get_ticker(symbol: str)`: This private static method will take a stock symbol as its argument. 
    It will retrieve the ticker for that symbol from Yahoo Finance using the `yf.Ticker` method. 
    If the ticker doesn't exist, it should raise a `ValueError`.

2. `retrieve_data(ticker_symbol: str, start_date = None, end_date = None)`: This static method will fetch the 
    stock data for a given symbol within the provided date range. If no date range is given, it should fetch all 
    available data.

3. `get_last_close_and_date(ticker_symbol: str)`: This static method will fetch the last closing price and its 
                                                  corresponding date for a given stock symbol.

**Steps**:

1. Use the provided blueprint of the `YahooFinanceDataLoader` class to begin your implementation. Add the @staticmethod
    to the 3 methods as this class is a utility class
2. Implement the `_get_ticker` method.
3. Implement the `retrieve_data` method.
4. Implement the `get_last_close_and_date` method.
5. Once you have completed your implementation, test the class by fetching data for some stock symbols. For instance:
    ```
    df = YahooFinanceDataLoader.retrieve_data("AAPL", start_date="2022-01-01", end_date="2023-01-01")
    print(df)

    date, close_price = YahooFinanceDataLoader.get_last_close_and_date("AAPL")
    print(f"Last closing date: {date}, Last closing price: {close_price}")
    ```

6. Ensure that your class handles potential errors.

**Hints**:

- `yf.Ticker()` provides the ability to get information about a stock by passing its symbol.
-  To test if a ticker exist and contains data you can try to perform this operation: ticker.info
- `ticker.history(period="1d", start=start_date, end=end_date)` is a method to fetch the stock data within a 
    specified date range. If no date range is provided, it fetches the maximum available data.
-  you can add a short description of the method in it to add clarity to your code

"""


class YahooFinanceDataLoader:

    @staticmethod
    def _get_ticker(symbol: str) -> yf.Ticker:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period='7d',interval='1d')
        if len(history) == 0:
            raise ValueError(f"Ticker: {ticker} does not exists.")
        return ticker

    @staticmethod
    def retrieve_data(ticker_symbol: str, start_date = None, end_date = None) -> pd.DataFrame:
        """
        Retrieve data by providing start_date or end_date or no dates or both.
        Date format : YYYY-MM-DD
        """
        ticker = YahooFinanceDataLoader._get_ticker(ticker_symbol)
        if start_date or end_date:
            return ticker.history(start=start_date, end=end_date, period="max")
        return ticker.history(period='max').v

    @staticmethod
    def get_last_close_and_date(ticker_symbol: str) -> tuple[datetime.date, float]:
        ticker = YahooFinanceDataLoader._get_ticker(ticker_symbol)
        data = ticker.history(period='7d',interval='1d')
        return data.index[-1].date(), data["Close"][-1]


"""
Exercise 2: Implement the `Position` class for Portfolio Management

In this exercise, you will be building the `Position` class, which will help in tracking the historical changes in 
weight and quantity for a given financial asset.

**Objective**: 
1. Implement the `Position` class, ensuring that you keep track of the historical weight and quantity of a financial asset.
2. Implement methods to update the weight and quantity of a financial asset and store these updates historically.

**Instructions**:
1. **Position Class**:
   - **Initialization**: The `__init__` method initializes an instance of the `Position` class. 
   It takes in a `financial_asset` (instance of the `FinancialAsset` class), an optional `weight`, 
   and an optional `quantity`. It also initializes empty dictionaries for `historical_weight` and `historical_quantity`.

   - **update_weight**:
     This method updates the current weight of the financial asset and stores the previous weight in the 
     `historical_weight` dictionary with the update time as the key. If no date is provided, use the current time.

   - **update_quantity**:
     Similarly, this method updates the current quantity of the financial asset and stores the previous quantity 
     in the `historical_quantity` dictionary with the update time as the key. If no date is provided, 
     use the current time.

2. **Test your implementation**:
   Once you've implemented the `Position` class, create an instance of it and test both the `update_weight` and
   `update_quantity` methods. Ensure that the historical dictionaries are updating as expected.

   Example:
   last_date, last_close = YahooFinanceDataLoader.get_last_close_and_date('AAPL')
    equity_last_quote = Quote(last_date, last_close)
    equity = Equity('AAPL', equity_last_quote, 'USD', 0.24)

    apple_position = Position(financial_asset=apple, weight=0.2, quantity=10)

   apple_position.update_weight(0.25)
   apple_position.update_quantity(15)

   print(apple_position.historical_weight)   # Expected output: {datetime_object: 0.2}
   print(apple_position.historical_quantity) # Expected output: {datetime_object: 10}

**Hints**:
- The `datetime.now()` method returns the current datetime, which can be used as a default when no update date is 
  provided.
"""


class Position:
    def __init__(self, financial_asset: FinancialAsset, weight: float = 0, quantity: float = 0):
        self.historical_weight: dict[datetime, float] = {}
        self.historical_quantity: dict[datetime, float] = {}
        self.financial_asset = financial_asset
        self.weight = None
        self.quantity = None
        self.update_weight(weight)
        self.update_quantity(quantity)

    def update_weight(self, weight: float, update_date: datetime = None):
        update_date = update_date if update_date else datetime.now()
        self.historical_weight[update_date] = weight
        self.weight = weight       

    def update_quantity(self, quantity: float, update_date: datetime = None):
        update_date = update_date if update_date else datetime.now()
        self.historical_quantity[update_date] = quantity     
        self.quantity = quantity       


"""
Exercise 3: Equity Portfolio Implementation Exercise:

**Objective:** 
You need to implement methods and functionalities for the `EquityPft` class. The class should handle the operations 
of initializing the portfolio with financial assets, rebalancing it based on a given strategy, 
and providing a summary of the portfolio positions.

**Instructions:**
1. **Class Attributes**:
   Based on the provided code framework, the `EquityPft` class should have the following attributes:
   - `name`: Name of the portfolio.
   - `code`: A unique code or identifier for the portfolio.
   - `currency`: The currency in which the portfolio is denominated.
   - `aum`: Assets Under Management.
   - `nb_of_shares`: Total number of shares in the portfolio.
   - `historical_NAV`: A list that keeps track of the historical Net Asset Values.
   - `positions`: A list of `Position` objects representing the portfolio's assets.
   - `strategy`: A strategy object, an instance of a class derived from the `Strategy` class.

2. **Initialization**:
   Implement the `__init__` method to initialize the portfolio attributes.

3. **Initialize Portfolio Positions**:
   Implement the `initialize_position_from_instrument_list` method to initialize the portfolio's positions with a given 
   list of `FinancialAsset` objects. Each financial asset should be wrapped in a `Position` object.

4. **Positions to Dictionary**: Already implemented
   Implement the `_positions_to_dict` method which returns a dictionary of positions with ticker symbols as keys and 
   `Position` objects as values. This will help in rebalancing operations.

5. **Rebalancing**:

   Implement the `rebalance_portfolio` method which performs the following tasks:
   - Convert the positions list into a dictionary.
   - Generate trading signals using the provided strategy.
   - Update the weight and quantity of each position in the portfolio based on the generated signals.

6. **Portfolio Summary**:
   Implement the `portfolio_position_summary` method. This method should return a dataframe summarizing the 
   portfolio's current positions, including the ticker symbols, weights, quantities, and last close prices of the assets.


**Tips**:
- Use the methods and attributes of other provided classes like `FinancialAsset`, `Quote`, and `Position` 
- Remember to handle cases where certain attributes might be `None` or missing.
- While rebalancing, consider how to handle assets that might not have a corresponding signal from the strategy.
"""

from abc import ABC, abstractmethod


class Strategy(ABC):

    @abstractmethod
    def generate_signals(self, data_for_signal_generation: dict):
        """
        Generate trading signals based on a series of prices.

        Parameters:
        - data_for_signal_generation: A dictionary with tickers as keys and positions as values.

        Returns:
        - A dictionary with tickers as keys and signals as values.
        """
        pass


class EqualWeightStrategy(Strategy):
    def generate_signals(self, position_dict: dict):
        tickers = position_dict.keys()
        equal_weight = 1 / len(tickers)
        return {ticker: equal_weight for ticker in tickers}


class EquityPft:
    def __init__(self, name: str, code: str, currency: str, aum: float, nb_of_shares: int, strategy: Strategy):
        self.name = name
        self.code = code
        self.currency = currency
        self.aum = aum
        self.nb_of_shares = nb_of_shares
        self.strategy = strategy()
        self.positions = []

    def _positions_to_dict(self):
        return {position.financial_asset.ticker: position for position in self.positions if position.weight is not None}

    def initialize_position_from_instrument_list(self, instrument_list: [FinancialAsset]):
        for instrument in instrument_list:
            self.positions.append(Position(instrument, 0, 0))

    def _reset_portfolio(self, position_dict: dict, rebalancing_date: datetime = None):
        for ticker, weight in position_dict.items():
            position_dict[ticker].update_weight(0, rebalancing_date)
            position_dict[ticker].update_quantity(0, rebalancing_date)

    def rebalance_portfolio(self, rebalancing_date: datetime = None):
        if not rebalancing_date:
            rebalancing_date = datetime.now()
        position_dict = self._positions_to_dict()
        self._reset_portfolio(position_dict, rebalancing_date)
        
        strategy = self.strategy.generate_signals(position_dict)
        for ticker, weight in strategy.items():
            last_quote = position_dict[ticker].financial_asset.last_quote.price
            quantity = self.aum * weight // last_quote
            position_dict[ticker].update_weight(weight, rebalancing_date)
            position_dict[ticker].update_quantity(quantity, rebalancing_date)

    def portfolio_position_summary(self) -> pd.DataFrame:
        # Preallocating lists is more efficient:
        len_positions = len(self.positions)
        indexes = ["weights", "quantities", "last_close_date", "last_close_price"]
        positions_dict = self._positions_to_dict()
        summary = {indexes_value: [None] * len_positions for indexes_value in indexes}

        for i, position in enumerate(self.positions):
            summary["weights"][i] = position.weight
            summary["quantities"][i] = position.quantity
            date, close = YahooFinanceDataLoader.get_last_close_and_date(position.financial_asset.ticker)
            summary["last_close_date"][i] = date
            summary["last_close_price"][i] = close
        df = pd.DataFrame(data=summary, index=list(positions_dict.keys()))
        return df


if __name__ == "__main__":

    quote_appl = Quote(datetime.now(), 100)
    quote_msft = Quote(datetime.now(), 200)
    quote_amzn = Quote(datetime.now(), 300)

    instrument_list = [
        FinancialAsset("AAPL", quote_appl, "USD"),
        FinancialAsset("MSFT", quote_msft, "USD"),
        FinancialAsset("AMZN", quote_amzn, "USD"),]

    pf = EquityPft(name="1", 
                   code=uuid4(), 
                   currency="USD", 
                   aum=1_000_000, 
                   nb_of_shares=100, 
                   strategy=EqualWeightStrategy)
    
    pf.initialize_position_from_instrument_list(instrument_list)
    pf.rebalance_portfolio()
    print(pf.portfolio_position_summary())