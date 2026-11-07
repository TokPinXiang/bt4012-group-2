---
name: get_transfers_between
description: Every transfer from one account to another, up to the alert time, oldest first.
handler: get_transfers_between
args:
  from_account:
    type: string
    description: Sender alias, e.g. A2
    pattern: "^A\\d+$"
  to_account:
    type: string
    description: Receiver alias, e.g. A9
    pattern: "^A\\d+$"
  reason:
    type: string
    description: One sentence on why you need these transfers and what you expect to learn. Logged for audit, not used by the lookup.
  window_hours:
    type: number
    description: How far back to look, in hours
    required: false
    default: 96
    minimum: 1
    maximum: 240
---
Use this to confirm one leg of a route (for a chain or cycle) or to see how often two accounts
trade. Transactions it returns get T aliases and can be cited.
