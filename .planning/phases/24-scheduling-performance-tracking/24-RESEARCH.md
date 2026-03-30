# Phase 24 Research

Scheduling + Performance Tracking

## SCHED-01: Monthly Rebalance
Celery Beat on 15th of month, pipeline: select_stocks -> compute_orders -> execute_rebalance

## SCHED-02: Stop-Loss Monitor
Every 5 min during TW trading hours (09:00-13:30 UTC+8), skip weekends

## SCHED-03: Daily NAV Snapshot
Post-close (14:00 UTC+8), nav_snapshots table, cash = initial_nav - costs

## PERF-01: Trade Log
trade_logs table, migration 012, entry/exit/pnl/holding_days

## PERF-02: Performance API
GET /api/portfolio/performance - NAV curve, return, drawdown, Sharpe

## PERF-03: Holdings API
GET /api/portfolio/holdings + GET /api/portfolio/orders

## Dependencies
Phase 22 (PortfolioStrategy), Phase 23 (OrderManager), Celery Beat, FastAPI
