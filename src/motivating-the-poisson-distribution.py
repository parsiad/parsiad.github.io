# %% [raw]
# +++
# aliases = [
#   "/blog/2019/motivating_the_poisson_distribution"
# ]
# date = 2019-12-01
# title = "Motivating the Poisson distribution"
# +++

# %% tags=["no_cell"]
import math

import matplotlib.pyplot as plt
import numpy as np

from _boilerplate import display_fig, init

init()

# %% [markdown]
# ## Introduction
#
# The Poisson distribution is usually introduced by its probability mass function (PMF).
# It is then tied back to the binomial distribution by showing that a particular parameterization of the binomial distribution converges to the Poisson distribution.
# This approach is somewhat unsatisfying: it does not give much insight into the Poisson distribution and, namely, why it is used to model certain phenomena.
#
# In this short post, we avoid motivating the Poisson distribution by its PMF and instead *construct* it directly from the binomial distribution.
#
# ## Prerequisites
#
# To understand this post, you will need to be familiar with the following concepts:
#
# * [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function) (PMF)
# * [independent and identically distributed](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) (IID)
# * [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
# * [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution)
#
# ## Construction by partitions
#
# Let's start with a motivating example: we would like to model the number of emails you receive during a particular work day (e.g., between 9am and 5pm).
# For convenience, we can normalize this interval of time by relabeling the start of the day 0 and the end 1.
#
# While an email may arrive at any time between 0 and 1, it will be easier to start with a model in which emails arrive only at one of finitely many times between 0 and 1.
# We do this by subdividing our day into $n$ uniformly sized partitions, each having length $h = 1 / n$:
#
# <img alt="" src="data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIHZpZXdCb3g9IjAuMCAwLjAgNDMyLjAgMTAwLjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InNxdWFyZSIgc3Ryb2tlLW1pdGVybGltaXQ9IjEwIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2xpcFBhdGggaWQ9InAuMCI+PHBhdGggZD0ibTAgMGw0MzIuMCAwbDAgMTAwLjBsLTQzMi4wIDBsMCAtMTAwLjB6IiBjbGlwLXJ1bGU9Im5vbnplcm8iLz48L2NsaXBQYXRoPjxnIGNsaXAtcGF0aD0idXJsKCNwLjApIj48cGF0aCBmaWxsPSIjMDAwMDAwIiBmaWxsLW9wYWNpdHk9IjAuMCIgZD0ibTAgMGw0MzIuMCAwbDAgMTAwLjBsLTQzMi4wIDB6IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiLz48cGF0aCBmaWxsPSIjMDAwMDAwIiBmaWxsLW9wYWNpdHk9IjAuMCIgZD0ibTE2LjAgNzIuMGw0MDAuMCAwIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiLz48cGF0aCBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMS4wIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbGluZWNhcD0iYnV0dCIgZD0ibTE2LjAgNzIuMGw0MDAuMCAwIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiLz48cGF0aCBmaWxsPSIjMDAwMDAwIiBmaWxsLW9wYWNpdHk9IjAuMCIgZD0ibTQxNi4wIDcyLjBsMCAtMjQuMCIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjEuMCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIGQ9Im00MTYuMCA3Mi4wbDAgLTI0LjAiIGZpbGwtcnVsZT0iZXZlbm9kZCIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGZpbGwtb3BhY2l0eT0iMC4wIiBkPSJtMTc2LjAgNzIuMGwwIC0yNC4wIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiLz48cGF0aCBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMS4wIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbGluZWNhcD0iYnV0dCIgZD0ibTE3Ni4wIDcyLjBsMCAtMjQuMCIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggZmlsbD0iIzAwMDAwMCIgZmlsbC1vcGFjaXR5PSIwLjAiIGQ9Im05Ni4wIDcyLjBsMCAtMjQuMCIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjEuMCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIGQ9Im05Ni4wIDcyLjBsMCAtMjQuMCIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggZmlsbD0iIzAwMDAwMCIgZmlsbC1vcGFjaXR5PSIwLjAiIGQ9Im0xNi4wIDcyLjBsMCAtMjQuMCIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjEuMCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIGQ9Im0xNi4wIDcyLjBsMCAtMjQuMCIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggZmlsbD0iIzAwMDAwMCIgZmlsbC1vcGFjaXR5PSIwLjAiIGQ9Im0xNi4wIDguMGwxNjAuMCAwbDAgNDAuMGwtMTYwLjAgMHoiIGZpbGwtcnVsZT0iZXZlbm9kZCIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGQ9Im05NC4zNjcxOSAyMS45NjY4NzNsMCA2LjEwOTM3NXExLjAxNTYyNSAtMS4xMjUgMS42MDkzNzUgLTEuNDM3NXEwLjU5Mzc1IC0wLjMxMjUgMS4xODc1IC0wLjMxMjVxMC43MDMxMjUgMCAxLjIwMzEyNSAwLjM5MDYyNXEwLjUxNTYyNSAwLjM5MDYyNSAwLjc2NTYyNSAxLjIzNDM3NXEwLjE3MTg3NSAwLjU3ODEyNSAwLjE3MTg3NSAyLjEyNWwwIDIuOTUzMTI1cTAgMC43OTY4NzUgMC4xMjUgMS4wOTM3NXEwLjA5Mzc1IDAuMjE4NzUgMC4zMTI1IDAuMzQzNzVxMC4yMTg3NSAwLjEyNSAwLjc5Njg3NSAwLjEyNWwwIDAuMzI4MTI1bC00LjA5Mzc1IDBsMCAtMC4zMjgxMjVsMC4xODc1IDBxMC41OTM3NSAwIDAuODEyNSAtMC4xNzE4NzVxMC4yMzQzNzUgLTAuMTg3NSAwLjMyODEyNSAtMC41MzEyNXEwLjAxNTYyNSAtMC4xNDA2MjUgMC4wMTU2MjUgLTAuODU5Mzc1bDAgLTIuOTUzMTI1cTAgLTEuMzU5Mzc1IC0wLjE0MDYyNSAtMS43ODEyNXEtMC4xNDA2MjUgLTAuNDM3NSAtMC40NTMxMjUgLTAuNjQwNjI1cS0wLjI5Njg3NSAtMC4yMTg3NSAtMC43MzQzNzUgLTAuMjE4NzVxLTAuNDUzMTI1IDAgLTAuOTM3NSAwLjIzNDM3NXEtMC40ODQzNzUgMC4yMzQzNzUgLTEuMTU2MjUgMC45NTMxMjVsMCA0LjQwNjI1cTAgMC44NTkzNzUgMC4wOTM3NSAxLjA3ODEyNXEwLjA5Mzc1IDAuMjAzMTI1IDAuMzU5Mzc1IDAuMzQzNzVxMC4yNjU2MjUgMC4xNDA2MjUgMC44OTA2MjUgMC4xNDA2MjVsMCAwLjMyODEyNWwtNC4xMjUgMGwwIC0wLjMyODEyNXEwLjU0Njg3NSAwIDAuODc1IC0wLjE3MTg3NXEwLjE3MTg3NSAtMC4wOTM3NSAwLjI4MTI1IC0wLjM0Mzc1cTAuMTA5Mzc1IC0wLjI2NTYyNSAwLjEwOTM3NSAtMS4wNDY4NzVsMCAtNy41NDY4NzVxMCAtMS40Mzc1IC0wLjA2MjUgLTEuNzY1NjI1cS0wLjA2MjUgLTAuMzI4MTI1IC0wLjIwMzEyNSAtMC40Mzc1cS0wLjE0MDYyNSAtMC4xMjUgLTAuMzkwNjI1IC0wLjEyNXEtMC4xODc1IDAgLTAuNjA5Mzc1IDAuMTU2MjVsLTAuMTI1IC0wLjMyODEyNWwyLjQ4NDM3NSAtMS4wMTU2MjVsMC40MjE4NzUgMHoiIGZpbGwtcnVsZT0ibm9uemVybyIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGZpbGwtb3BhY2l0eT0iMC4wIiBkPSJtLTY0LjAgOC4wbDE2MC4wIDBsMCA0MC4wbC0xNjAuMCAweiIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggZmlsbD0iIzAwMDAwMCIgZD0ibTEyLjAwNzgxMjUgMjguODEwNjIzcTAgLTIuMTA5Mzc1IDAuNjQwNjI1IC0zLjYyNXEwLjY0MDYyNSAtMS41MzEyNSAxLjY4NzUgLTIuMjgxMjVxMC44MjgxMjUgLTAuNTkzNzUgMS43MDMxMjUgLTAuNTkzNzVxMS40MjE4NzUgMCAyLjU0Njg3NSAxLjQ1MzEyNXExLjQwNjI1IDEuNzk2ODc1IDEuNDA2MjUgNC44NTkzNzVxMCAyLjE1NjI1IC0wLjYyNSAzLjY1NjI1cS0wLjYwOTM3NSAxLjUgLTEuNTc4MTI1IDIuMTg3NXEtMC45NTMxMjUgMC42NzE4NzUgLTEuODQzNzUgMC42NzE4NzVxLTEuNzY1NjI1IDAgLTIuOTM3NSAtMi4wOTM3NXEtMS4wIC0xLjc1IC0xLjAgLTQuMjM0Mzc1em0xLjc4MTI1IDAuMjM0Mzc1cTAgMi41NDY4NzUgMC42NDA2MjUgNC4xNTYyNXEwLjUxNTYyNSAxLjM1OTM3NSAxLjU0Njg3NSAxLjM1OTM3NXEwLjQ4NDM3NSAwIDEuMDE1NjI1IC0wLjQzNzVxMC41MzEyNSAtMC40Mzc1IDAuNzk2ODc1IC0xLjQ4NDM3NXEwLjQyMTg3NSAtMS41NjI1IDAuNDIxODc1IC00LjQyMTg3NXEwIC0yLjEwOTM3NSAtMC40Mzc1IC0zLjUxNTYyNXEtMC4zMjgxMjUgLTEuMDQ2ODc1IC0wLjg0Mzc1IC0xLjQ4NDM3NXEtMC4zNzUgLTAuMjk2ODc1IC0wLjkwNjI1IC0wLjI5Njg3NXEtMC42MDkzNzUgMCAtMS4wOTM3NSAwLjU0Njg3NXEtMC42NTYyNSAwLjc2NTYyNSAtMC45MDYyNSAyLjM5MDYyNXEtMC4yMzQzNzUgMS42MDkzNzUgLTAuMjM0Mzc1IDMuMTg3NXoiIGZpbGwtcnVsZT0ibm9uemVybyIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGZpbGwtb3BhY2l0eT0iMC4wIiBkPSJtOTYuMCA4LjBsMTYwLjAgMGwwIDQwLjBsLTE2MC4wIDB6IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiLz48cGF0aCBmaWxsPSIjMDAwMDAwIiBkPSJtMTc1LjIxODc1IDMyLjU0NWwtMC44NTkzNzUgMi4zNzVsLTcuMjgxMjUgMGwwIC0wLjM0Mzc1cTMuMjAzMTI1IC0yLjkyMTg3NSA0LjUxNTYyNSAtNC43ODEyNXExLjMxMjUgLTEuODU5Mzc1IDEuMzEyNSAtMy40MDYyNXEwIC0xLjE3MTg3NSAtMC43MTg3NSAtMS45MjE4NzVxLTAuNzE4NzUgLTAuNzY1NjI1IC0xLjcxODc1IC0wLjc2NTYyNXEtMC45MDYyNSAwIC0xLjY0MDYyNSAwLjU0Njg3NXEtMC43MTg3NSAwLjUzMTI1IC0xLjA2MjUgMS41NDY4NzVsLTAuMzQzNzUgMHEwLjIzNDM3NSAtMS42NzE4NzUgMS4xNzE4NzUgLTIuNTc4MTI1cTAuOTUzMTI1IC0wLjkwNjI1IDIuMzU5Mzc1IC0wLjkwNjI1cTEuNSAwIDIuNSAwLjk2ODc1cTEuMDE1NjI1IDAuOTY4NzUgMS4wMTU2MjUgMi4yODEyNXEwIDAuOTM3NSAtMC40Mzc1IDEuODc1cS0wLjY3MTg3NSAxLjQ2ODc1IC0yLjE4NzUgMy4xMjVxLTIuMjY1NjI1IDIuNDY4NzUgLTIuODI4MTI1IDIuOTg0Mzc1bDMuMjE4NzUgMHEwLjk4NDM3NSAwIDEuMzc1IC0wLjA2MjVxMC40MDYyNSAtMC4wNzgxMjUgMC43MTg3NSAtMC4yOTY4NzVxMC4zMjgxMjUgLTAuMjM0Mzc1IDAuNTYyNSAtMC42NDA2MjVsMC4zMjgxMjUgMHptMy44MTI1IC0xMC41NzgxMjVsMCA2LjEwOTM3NXExLjAxNTYyNSAtMS4xMjUgMS42MDkzNzUgLTEuNDM3NXEwLjU5Mzc1IC0wLjMxMjUgMS4xODc1IC0wLjMxMjVxMC43MDMxMjUgMCAxLjIwMzEyNSAwLjM5MDYyNXEwLjUxNTYyNSAwLjM5MDYyNSAwLjc2NTYyNSAxLjIzNDM3NXEwLjE3MTg3NSAwLjU3ODEyNSAwLjE3MTg3NSAyLjEyNWwwIDIuOTUzMTI1cTAgMC43OTY4NzUgMC4xMjUgMS4wOTM3NXEwLjA5Mzc1IDAuMjE4NzUgMC4zMTI1IDAuMzQzNzVxMC4yMTg3NSAwLjEyNSAwLjc5Njg3NSAwLjEyNWwwIDAuMzI4MTI1bC00LjA5Mzc1IDBsMCAtMC4zMjgxMjVsMC4xODc1IDBxMC41OTM3NSAwIDAuODEyNSAtMC4xNzE4NzVxMC4yMzQzNzUgLTAuMTg3NSAwLjMyODEyNSAtMC41MzEyNXEwLjAxNTYyNSAtMC4xNDA2MjUgMC4wMTU2MjUgLTAuODU5Mzc1bDAgLTIuOTUzMTI1cTAgLTEuMzU5Mzc1IC0wLjE0MDYyNSAtMS43ODEyNXEtMC4xNDA2MjUgLTAuNDM3NSAtMC40NTMxMjUgLTAuNjQwNjI1cS0wLjI5Njg3NSAtMC4yMTg3NSAtMC43MzQzNzUgLTAuMjE4NzVxLTAuNDUzMTI1IDAgLTAuOTM3NSAwLjIzNDM3NXEtMC40ODQzNzUgMC4yMzQzNzUgLTEuMTU2MjUgMC45NTMxMjVsMCA0LjQwNjI1cTAgMC44NTkzNzUgMC4wOTM3NSAxLjA3ODEyNXEwLjA5Mzc1IDAuMjAzMTI1IDAuMzU5Mzc1IDAuMzQzNzVxMC4yNjU2MjUgMC4xNDA2MjUgMC44OTA2MjUgMC4xNDA2MjVsMCAwLjMyODEyNWwtNC4xMjUgMGwwIC0wLjMyODEyNXEwLjU0Njg3NSAwIDAuODc1IC0wLjE3MTg3NXEwLjE3MTg3NSAtMC4wOTM3NSAwLjI4MTI1IC0wLjM0Mzc1cTAuMTA5Mzc1IC0wLjI2NTYyNSAwLjEwOTM3NSAtMS4wNDY4NzVsMCAtNy41NDY4NzVxMCAtMS40Mzc1IC0wLjA2MjUgLTEuNzY1NjI1cS0wLjA2MjUgLTAuMzI4MTI1IC0wLjIwMzEyNSAtMC40Mzc1cS0wLjE0MDYyNSAtMC4xMjUgLTAuMzkwNjI1IC0wLjEyNXEtMC4xODc1IDAgLTAuNjA5Mzc1IDAuMTU2MjVsLTAuMTI1IC0wLjMyODEyNWwyLjQ4NDM3NSAtMS4wMTU2MjVsMC40MjE4NzUgMHoiIGZpbGwtcnVsZT0ibm9uemVybyIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGZpbGwtb3BhY2l0eT0iMC4wIiBkPSJtMjE2LjAgOC4wbDE2MC4wIDBsMCA0MC4wbC0xNjAuMCAweiIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggZmlsbD0iIzAwMDAwMCIgZD0ibTI5MS4zMzIwMyAzMy4xNTQzNzNxMC40Mzc1IDAgMC43MTg3NSAwLjI5Njg3NXEwLjI5Njg3NSAwLjI5Njg3NSAwLjI5Njg3NSAwLjcxODc1cTAgMC40MDYyNSAtMC4yOTY4NzUgMC43MDMxMjVxLTAuMjk2ODc1IDAuMjk2ODc1IC0wLjcxODc1IDAuMjk2ODc1cS0wLjQyMTg3NSAwIC0wLjcxODc1IC0wLjI5Njg3NXEtMC4yODEyNSAtMC4yOTY4NzUgLTAuMjgxMjUgLTAuNzAzMTI1cTAgLTAuNDM3NSAwLjI4MTI1IC0wLjcxODc1cTAuMjk2ODc1IC0wLjI5Njg3NSAwLjcxODc1IC0wLjI5Njg3NXptNC42NjQwNjI1IDBxMC40Mzc1IDAgMC43MTg3NSAwLjI5Njg3NXEwLjI5Njg3NSAwLjI5Njg3NSAwLjI5Njg3NSAwLjcxODc1cTAgMC40MDYyNSAtMC4yOTY4NzUgMC43MDMxMjVxLTAuMjk2ODc1IDAuMjk2ODc1IC0wLjcxODc1IDAuMjk2ODc1cS0wLjQyMTg3NSAwIC0wLjcxODc1IC0wLjI5Njg3NXEtMC4yODEyNSAtMC4yOTY4NzUgLTAuMjgxMjUgLTAuNzAzMTI1cTAgLTAuNDM3NSAwLjI4MTI1IC0wLjcxODc1cTAuMjk2ODc1IC0wLjI5Njg3NSAwLjcxODc1IC0wLjI5Njg3NXptNC42NjQwNjI1IDBxMC40Mzc1IDAgMC43MTg3NSAwLjI5Njg3NXEwLjI5Njg3NSAwLjI5Njg3NSAwLjI5Njg3NSAwLjcxODc1cTAgMC40MDYyNSAtMC4yOTY4NzUgMC43MDMxMjVxLTAuMjk2ODc1IDAuMjk2ODc1IC0wLjcxODc1IDAuMjk2ODc1cS0wLjQyMTg3NSAwIC0wLjcxODc1IC0wLjI5Njg3NXEtMC4yODEyNSAtMC4yOTY4NzUgLTAuMjgxMjUgLTAuNzAzMTI1cTAgLTAuNDM3NSAwLjI4MTI1IC0wLjcxODc1cTAuMjk2ODc1IC0wLjI5Njg3NSAwLjcxODc1IC0wLjI5Njg3NXoiIGZpbGwtcnVsZT0ibm9uemVybyIvPjxwYXRoIGZpbGw9IiMwMDAwMDAiIGZpbGwtb3BhY2l0eT0iMC4wIiBkPSJtMzM2LjAgOC4wbDE2MC4wIDBsMCA0MC4wbC0xNjAuMCAweiIgZmlsbC1ydWxlPSJldmVub2RkIi8+PHBhdGggZmlsbD0iIzAwMDAwMCIgZD0ibTQxMy41MjM0NCAyMy43NzkzNzNsMy4wIC0xLjQ2ODc1bDAuMzEyNSAwbDAgMTAuNDM3NXEwIDEuMDMxMjUgMC4wNzgxMjUgMS4yOTY4NzVxMC4wOTM3NSAwLjI1IDAuMzU5Mzc1IDAuMzkwNjI1cTAuMjgxMjUgMC4xMjUgMS4xMDkzNzUgMC4xNDA2MjVsMCAwLjM0Mzc1bC00LjY0MDYyNSAwbDAgLTAuMzQzNzVxMC44NzUgLTAuMDE1NjI1IDEuMTI1IC0wLjE0MDYyNXEwLjI2NTYyNSAtMC4xNDA2MjUgMC4zNTkzNzUgLTAuMzU5Mzc1cTAuMDkzNzUgLTAuMjE4NzUgMC4wOTM3NSAtMS4zMjgxMjVsMCAtNi42NzE4NzVxMCAtMS4zNDM3NSAtMC4wNzgxMjUgLTEuNzM0Mzc1cS0wLjA3ODEyNSAtMC4yOTY4NzUgLTAuMjUgLTAuNDIxODc1cS0wLjE1NjI1IC0wLjE0MDYyNSAtMC4zOTA2MjUgLTAuMTQwNjI1cS0wLjM0Mzc1IDAgLTAuOTM3NSAwLjI4MTI1bC0wLjE0MDYyNSAtMC4yODEyNXoiIGZpbGwtcnVsZT0ibm9uemVybyIvPjwvZz48L3N2Zz4=" />
#
# We assume that inside each of these partitions, you receive at most one email.
# Let $X_k = 1$ if you received an email between $(k-1)h$ and $kh$.
# Otherwise, let $X_k = 0$.
#
# Moreover, we assume that emails are independent (while this might be a lofty assumption, we leave it to the reader to create more complicated models) and that the probability of receiving an email in one partition is identical to that of receiving an email in another partition.
# These assumptions can be neatly summarized in one sentence: the random variables $X_1, \ldots, X_n$ are IID.
#
# The sum $S_n = X_1 + \cdots + X_n$ counts the total number of emails received in the work day.
# Being a sum of IID Bernoulli random variables, $S_n$ has binomial distribution with $n$ trials and success probability $p = \mathbb{P}(X_1 = 1)$.
# That is, $S_n$ has PMF
#
# $$
#   f_{n, p}(s) = \binom{n}{s} p^s \left( 1 - p \right)^{n-s}.
# $$
#
# The expected number of emails received under this model is $\mathbb{E} S_n = np$.
# We would like to pick $p$ such that the expected number of emails does not depend on the number of partitions $n$.
# Our model would be a bit weird if the expected number of emails changed as a function of the number of partitions!
# The only way to prevent this from happening is to pick $p = \lambda / n$ for some positive constant $\lambda$
# (technically, it's possible that $p = \lambda / n > 1$, but we can always pick $n$ sufficiently large to obtain a valid probability).
# Under this choice, the PMF becomes
#
# $$
#   f_{n,\lambda/n}(s)
#   = \binom{n}{s}\left(\frac{\lambda}{n}\right)^{s}\left(1-\frac{\lambda}{n}\right)^{n-s}
#   = \frac{\lambda^{s}}{s!}\frac{1}{n^{n}}\frac{n!}{\left(n-s\right)!}\left(n-\lambda\right)^{n-s}.
# $$
#
# Next, note that
#
# $$
# \begin{aligned}
#   \frac{1}{n^{n}}\frac{n!}{\left(n-s\right)!}\left(n-\lambda\right)^{n-s}
#   & = \frac{1}{n^{n}}\frac{n!}{\left(n-s\right)!}\sum_{k=0}^{n-s}\binom{n-s}{k}\left(-\lambda\right)^{k}n^{n-s-k} \\\\
#   & = \sum_{k=0}^{n-s}\frac{\left(-\lambda\right)^{k}}{k!}\frac{n\left(n-1\right)\cdots\left(n-s-k+1\right)}{n^{s+k}}.
# \end{aligned}
# $$
#
# As we increase the number of partitions, our model becomes more and more realistic.
# By taking a limit as the number of partitions goes to infinity, we obtain a model in which emails can be received at any point in time.
# We make a new PMF $g_\lambda$ by taking this limit:
#
# $$
#   g_\lambda(s)
#   = \lim_{n}f_{n,\lambda/n}(s)
#   = \frac{\lambda^s}{s!} \sum_{k \geq 0} \frac{\left(-\lambda\right)^k}{k!}
#   = \frac{\lambda^{s}}{s!}e^{-\lambda}.
# $$
#
# The figure below plots $g_\lambda$ as a function of $s$.
# When $\lambda$ is a positive integer, the modes are $\lambda$ and $\lambda - 1$, as can be seen in the plot.

# %% tags=["no_input"]
fig = plt.figure()
s = np.arange(25 + 1)
for r in [1.0, 5.0, 10.0, 15.0]:
    g = r**s / np.array([math.factorial(si) for si in s]) * np.exp(-r)
    plt.plot(s, g, ":o", label=rf"$\lambda = {r}$")
plt.xlabel("$s$")
plt.ylabel(r"$g_\lambda(s)$")
plt.legend()
_ = plt.title("Poisson probability mass function")
display_fig(fig)

# %% [markdown]
# Our construction above suggests that the PMF $g_\lambda$ models the number of emails received per work day assuming that
#
# * the receipt (or non-receipt) of one email does not affect future emails and
# * the expected number of emails received is specified by the parameter $\lambda$.
#
# Of course, we can replace "email" by any event and "work day" by any finite time horizon we are interested in, so long as the modeling assumptions above are sound.
#
# In summary, any random variable with PMF $g_\lambda$ (for some positive value of $\lambda$) is said to be a *Poisson random variable* (or, equivalently, have a *Poisson distribution*).
# All of our hard work above is not only a construction of the Poisson distribution but also a proof of the following result:
#
# **Proposition (Binomial to Poisson).**
# Let $\lambda$ be a positive number.
# The binomial distribution with $n$ trials and success probability $p = \lambda / n$ [converges](https://en.wikipedia.org/wiki/Convergence_of_random_variables#Convergence_in_distribution), as $n \rightarrow \infty$, to the Poisson distribution with parameter $\lambda$.
#
# ## The usual rigmarole
#
# For completeness, we also give the approach alluded to in the introduction of this article.
# In particular, we take the definition of the Poisson distribution involving $g_\lambda$ as given and prove the proposition of the previous section using a more expedient approach.
#
# We establish the proposition by showing that the [characteristic function](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)) (CF) of the binomial distribution conveges to that of the Poisson distribution and applying [Lévy's continuity theorem](https://en.wikipedia.org/wiki/L%C3%A9vy%27s_continuity_theorem).
#
# *Proof (Binomial to Poisson)*.
# Let's start by computing the CF of the Poisson distribution:
#
# $$
#     \varphi_{\mathrm{Poisson}(\lambda)}(t)
#     = e^{-\lambda}\sum_{k\geq0}\frac{1}{k!}\left(\lambda e^{it}\right)^{k}
#     = e^{-\lambda}\exp\left(\lambda e^{it}\right)
#     = \exp\left(\lambda \left(e^{it} - 1\right)\right).
# $$
#
# Let's now compute the CF of the binomial distribution with $n$ trials and success probability $p$.
# The CF can be obtained by applying the [binomial theorem](https://en.wikipedia.org/wiki/Binomial_theorem):
#
# $$
# \begin{aligned}
#     \varphi_{\mathrm{Binomial}(n, p)}(t)
#     & = \sum_{k\geq0}e^{itk}\binom{n}{k}p^{k}\left(1-p\right)^{n-k} \\\\
#     & = \sum_{k\geq0}\binom{n}{k}\left(pe^{it}\right)^{k}\left(1-p\right)^{n-k} \\\\
#     & = \left(1-p+pe^{it}\right)^{n} \\\\
#     & = \left(1+p\left(e^{it}-1\right)\right)^{n}.
# \end{aligned}
# $$
#
# Setting $p = \lambda / n$ and taking limits in the above, we obtain the desired result:
#
# $$
#   \lim_n \varphi_{\mathrm{Binomial}(n, \lambda / n)}(t)
#   = \lim_n \left(
#     \exp \left( \frac{\lambda}{n} \left( e^{it} - 1 \right) \right)
#     + O \left( \frac{1}{n^2} \right)
#   \right)^n
#   = \varphi_{\mathrm{Poisson}(\lambda)}(t).
# $$
