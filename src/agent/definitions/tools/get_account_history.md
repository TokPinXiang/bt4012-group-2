---
name: get_account_history
description: Totals and most recent transactions of one account, up to the alert time.
handler: get_account_history
args:
  account:
    type: string
    description: Account alias from the case, e.g. A7
    pattern: "^A\\d+$"
  reason:
    type: string
    description: One sentence on why you need this account's history and what you expect to learn. Logged for audit, not used by the lookup.
  window_hours:
    type: number
    description: How far back to look, in hours
    required: false
    default: 96
    minimum: 1
    maximum: 240
  limit:
    type: integer
    description: Maximum number of recent transactions to list
    required: false
    default: 25
    minimum: 1
    maximum: 50
---
Use this to see what an intermediary account did before the alert: who funded it, where it sent
money, and whether it has a long normal history. Transactions it returns get T aliases and can be
cited. Nothing after the alert time is ever returned.
