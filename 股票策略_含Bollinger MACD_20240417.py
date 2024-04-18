# -*- encoding: utf-8 -*-
#@Author      : RayNG
#@Time        : 2024/04/17 15:03
#@File        : 股票策略_含Bollinger MACD_20240417.py

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
"""

"""

INDEX_PATH = r"E:\Quant_New\主图_clean\hs300.csv"
DIR_STOCK = r"E:\Quant_New\clean"
DIR_RESULT = r"E:\Quant_New\results"


class StockStrategy:

    def __init__(self, initial_balance: float = 1000000) -> None:
        #
        self.initial_balance = initial_balance
        #
        self.commission_rate = 0
        self.stamp_tax = 0
        self.transfer_fee = 0
        self.df_stock = pd.DataFrame()
        pass

    def set_commission(self,
                       commission_rate: float = 0.00025,
                       stamp_tax: float = 0.001,
                       transfer_fee: float = 0.00001):
        # 佣金，买卖都要收, 0.025%
        self.commission_rate = commission_rate
        # 印花税，仅卖出时收取, 0.1%
        self.stamp_tax = stamp_tax
        # 过户费，买卖都要收, 0.001%
        self.transfer_fee = transfer_fee

    def get_benchmark(self,
                      start_date: str = '2015-01-01',
                      end_date: str = '2023-12-31'):
        """
        获取主图, 并借此获得市场交易日日期
        """
        df_index = pd.read_csv(INDEX_PATH, encoding='utf-8')

        # 设置index为日期, 并设置成datetime, 方便后续引用
        df_index.index = df_index['date']
        pd.to_datetime(df_index.index)

        # 仅仅保留价格信息
        df_index = df_index[start_date:end_date][[
            'open', 'high', 'low', 'close'
        ]]
        return df_index

    def get_single_stock_historical_data(
        self,
        stock_code: str,
        start_date: str = '2015-01-01',
        end_date: str = '2023-12-31',
    ):
        data_path = os.path.join(DIR_STOCK, f'{stock_code}.csv')
        df_stock = pd.read_csv(data_path, encoding='utf-8', index_col=['date'])
        df_stock = df_stock[start_date:end_date]
        df_stock = df_stock[['close']]

        # 归一
        df_stock = self.value_initialize(df_stock, 'close', '股票指数')

        return df_stock

    def cal_macd(self,
                 df_stock: pd.DataFrame,
                 short_term: int = 5,
                 long_term: int = 26):
        short_ma = df_stock['close'].rolling(short_term).mean()
        long_ma = df_stock['close'].rolling(long_term).mean()
        df_stock['short_ma'] = short_ma
        df_stock['long_ma'] = long_ma
        df_stock['diff'] = short_ma - long_ma
        df_stock['diff_1'] = df_stock['diff'].shift(1)
        df_stock['execution'] = None
        for date in df_stock.index:
            # MACD从负变正，金叉，买入信号
            if df_stock.loc[date, 'diff_1'] < 0 and df_stock.loc[date,
                                                                 'diff'] > 0:
                df_stock.loc[date, 'execution'] = -1
            # MACD从正变负，死叉，卖出信号
            elif df_stock.loc[date, 'diff_1'] > 0 and df_stock.loc[date,
                                                                   'diff'] < 0:
                df_stock.loc[date, 'execution'] = 1
            else:
                df_stock.loc[date, 'execution'] = 0

        return df_stock[[
            'close', '股票指数', 'short_ma', 'long_ma', 'diff', 'execution'
        ]]

    def cal_bollinger(self,
                      df_stock: pd.DataFrame,
                      rolling_window: int = 20,
                      num_of_std: int = 2):

        blg_mean = df_stock['close'].rolling(rolling_window).mean()
        blg_std = df_stock['close'].rolling(rolling_window).std()

        df_stock['upper'] = blg_mean + num_of_std * blg_std
        df_stock['mean'] = blg_mean
        df_stock['lower'] = blg_mean - num_of_std * blg_std

        # 卖出记为1，买入记为-1，其余为0
        df_stock['execution'] = np.where(
            df_stock['close'] < df_stock['lower'], 1,
            np.where(df_stock['close'] > df_stock['upper'], -1, 0))
        return df_stock

    def execution(self, df_stock: pd.DataFrame):
        account_balance = self.initial_balance
        df_stock.fillna(0)

        # 初始化
        num_of_holding = 0
        execution_type = '--'

        # 记录
        list_execution = []
        list_num_of_execution = []
        list_holding = []
        list_account_balance = []
        list_fund_mv = []

        for date in df_stock.index:
            execution_set = df_stock.loc[date, 'execution']
            close_price = df_stock.loc[date, 'close']
            if execution_set == -1:
                num_to_buy = (account_balance /
                              (close_price *
                               (1 + self.commission_rate + self.transfer_fee))
                              // 100) * 100  # 先直接考虑佣金问题
                buy_cost = num_to_buy * close_price * (
                    1 + self.commission_rate + self.transfer_fee)
                if account_balance - buy_cost < 0:
                    num_to_buy = 0
                    buy_cost = 0
                account_balance -= buy_cost
                num_of_holding += num_to_buy
                # 记录交易类型
                if num_to_buy != 0:
                    execution_type = 'buy'
                list_num_of_execution.append(num_to_buy)
            elif execution_set == 1 and num_of_holding != 0:
                num_to_sell = num_of_holding
                sell_gain = num_to_sell * close_price * (
                    1 - self.commission_rate - self.stamp_tax -
                    self.transfer_fee)
                account_balance += sell_gain
                num_of_holding = 0
                execution_type = 'sell'
                # 记录交易类型
                list_num_of_execution.append(num_to_sell)
            else:
                list_num_of_execution.append('--')

            stock_holding_value = num_of_holding * close_price
            fund_mv = stock_holding_value + account_balance

            # 记录
            list_execution.append(execution_type)
            list_holding.append(num_of_holding)
            list_account_balance.append(account_balance)
            list_fund_mv.append(fund_mv)
            # 初始化以进行下一次循环
            num_to_buy = 0
            num_to_sell = 0
            execution_type = '--'

        df_result = pd.DataFrame([
            list_execution, list_holding, df_stock['close'],
            list_account_balance, list_fund_mv
        ]).T
        df_result.index = df_stock.index
        df_result.columns = ['交易类型', '持仓数量', '收盘价格', '账户余额', '基金市值']

        self.df_stock = pd.concat([df_stock, df_result], axis=1)

    def value_initialize(self, df: pd.DataFrame, col_name: str,
                         result_col_name: str):
        """
        首期归一
        """
        first_term = df.index[0]
        first_price = df.loc[first_term, col_name]
        df[result_col_name] = df[col_name] / first_price
        return df

    def get_trade_record(self, df_stock: pd.DataFrame):
        df_execution_index = df_stock['交易类型'].isin(['buy', 'sell'])
        df_execution = df_stock[df_execution_index][['交易类型', '持仓数量', '收盘价格']]
        return df_execution

    def get_win_rate(self, df_stock: pd.DataFrame, df_index: pd.DataFrame,
                     df_result: pd.DataFrame):
        """
        策略收益率, 策略年化收益率, 超额收益, 基准收益, Alpha, Beta, Sharpe Ratio, Win rate, 最大回撤
        """
        first_term = df_stock.index[0]
        last_term = df_stock.index[-1]
        # 策略收益率
        fund_profit_rate = df_stock.loc[last_term, '基金市值'] / df_stock.loc[
            first_term, '基金市值'] - 1
        annulized_fund_profit_rate = fund_profit_rate / (
            len(df_stock[first_term:last_term]) / 365)

        # 基准收益率
        index_profit_rate = df_index.loc[last_term, 'close'] / df_index.loc[
            first_term, 'close'] - 1

        # 超额收益率
        excess_profit_rate = df_result.loc[last_term, '基金指数'] / df_result.loc[
            last_term, '基准指数'] - 1

        # win rate
        win_rate = len(df_result[df_result['基金指数'] > df_result['基准指数']]) / len(
            df_result['基准指数'])

        df_profit_detail = pd.DataFrame(
            [
                fund_profit_rate, annulized_fund_profit_rate,
                index_profit_rate, excess_profit_rate, win_rate
            ],
            index=['策略收益率', '策略年化收益率', '基准收益率', '超额收益率', '胜率']).T
        return df_profit_detail

    def main_bollinger(self,
                       stock_code: str,
                       start_date: str = '2015-01-01',
                       end_date: str = '2023-12-31'):
        # 获得基准数据
        df_index = self.get_benchmark(start_date, end_date)
        # 获得单股票数据
        df_stock = self.get_single_stock_historical_data(
            stock_code, start_date, end_date)
        # 计算布林带
        df_stock_with_blg = self.cal_bollinger(df_stock)
        self.execution(df_stock_with_blg)
        # 提取重要数据
        df_execution = self.df_stock[['交易类型', '持仓数量', '收盘价格', '账户余额', '基金市值']]
        # 首期归一
        df_index = self.value_initialize(df_index, 'close', '基准指数')
        df_execution = self.value_initialize(df_execution, '基金市值', '基金指数')
        # 获取交易记录
        df_record = self.get_trade_record(df_execution)
        # 合并数据
        df_result = pd.concat(
            [df_execution[['基金指数']], df_index[['基准指数']], df_stock['股票指数']],
            axis=1).dropna()
        # 胜率数据
        df_profit_detail = self.get_win_rate(df_execution, df_index, df_result)
        return df_index, df_stock_with_blg, df_execution, df_record, df_result, df_profit_detail

    def main_macd(self,
                  stock_code: str,
                  start_date: str = '2015-01-01',
                  end_date: str = '2023-12-31'):
        # 获得基准数据
        df_index = self.get_benchmark(start_date, end_date)
        # 获得单股票数据
        df_stock = self.get_single_stock_historical_data(
            stock_code, start_date, end_date)
        df_stock_with_macd = self.cal_macd(df_stock)
        self.execution(df_stock_with_macd)
        # 提取重要数据
        df_execution = self.df_stock[['交易类型', '持仓数量', '收盘价格', '账户余额', '基金市值']]
        # 首期归一
        df_index = self.value_initialize(df_index, 'close', '基准指数')
        df_execution = self.value_initialize(df_execution, '基金市值', '基金指数')
        # 获取交易记录
        df_record = self.get_trade_record(df_execution)
        # 合并数据
        df_result = pd.concat(
            [df_execution[['基金指数']], df_index[['基准指数']], df_stock['股票指数']],
            axis=1).dropna()
        # 胜率数据
        df_profit_detail = self.get_win_rate(df_execution, df_index, df_result)
        return df_index, df_stock_with_macd, df_execution, df_record, df_result, df_profit_detail


if __name__ == '__main__':

    start_time = time.time()
    # 创建对象
    stock_strategy = StockStrategy()
    # 设置佣金
    stock_strategy.set_commission()
    # 单股票数据
    stock_code = 'hs300'
    # 起始日与截止日
    start_date = '2023-04-01'
    end_date = '2024-03-31'

    # ----------------------- * 布林带策略 * ----------------------- #
    df_index, df_stock_with_blg, df_execution, df_record, df_result, df_profit_detail = stock_strategy.main_bollinger(
        stock_code, start_date, end_date)

    end_time = time.time()

    print(f'\n计算Bollinger策略用时{round(end_time-start_time,2)}s\n正在导出...\n')

    with pd.ExcelWriter(
            os.path.join(
                DIR_RESULT, '{}_{}_{}.xlsx'.format(
                    stock_code, 'Bollinger',
                    datetime.now().strftime('%Y%m%d%H%M%S')))) as f:
        df_index.to_excel(f, sheet_name='Index Price', index=True)
        df_stock_with_blg.to_excel(f, sheet_name='Stock Price', index=True)
        df_execution.to_excel(f, sheet_name='Fund Record', index=True)
        df_record.to_excel(f, sheet_name='Trade Record', index=True)
        df_result.to_excel(f, 'Fund Value', index=True)
        df_profit_detail.to_excel(f, 'Return Detail', index=False)
    # ----------------------- * 布林带策略 * ----------------------- #

    # ----------------------- * MACD策略 * ----------------------- #
    start_time = time.time()

    df_index, df_stock_with_macd, df_execution, df_record, df_result, df_profit_detail = stock_strategy.main_macd(
        stock_code, start_date, end_date)

    end_time = time.time()
    print(f'\n计算MACD策略用时{round(end_time-start_time,2)}s\n正在导出...\n')

    with pd.ExcelWriter(
            os.path.join(
                DIR_RESULT, '{}_{}_{}.xlsx'.format(
                    stock_code, 'MACD',
                    datetime.now().strftime('%Y%m%d%H%M%S')))) as f:
        df_index.to_excel(f, sheet_name='Index Price', index=True)
        df_stock_with_macd.to_excel(f, sheet_name='Stock Price', index=True)
        df_execution.to_excel(f, sheet_name='Fund Record', index=True)
        df_record.to_excel(f, sheet_name='Trade Record', index=True)
        df_result.to_excel(f, 'Fund Value', index=True)
        df_profit_detail.to_excel(f, 'Return Detail', index=False)
    # ----------------------- * MACD策略 * ----------------------- #
