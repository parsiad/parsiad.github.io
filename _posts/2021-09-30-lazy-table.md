---
layout: post
title: lazy-table - A python-tabulate wrapper for producing tables from generators.
date: 2021-09-30 12:00:00-0800
---

I made a tiny [python-tabulate](https://github.com/astanin/python-tabulate) wrapper for producing tables from generators.
It's called lazy_table, and it even has a fancy logo:

![](https://raw.githubusercontent.com/parsiad/lazy-table/master/logo.png)

lazy_table is useful when (i) each row of your table is generated by a possibly expensive computation and (ii) you want to print rows to the screen as soon as they are available.
For example, the rows in the table below correspond to numerically solving an ordinary differential equation with progressively more steps.

![](https://raw.githubusercontent.com/parsiad/lazy-table/main/examples/euler_vdp.gif)

You can install it via pip:

```shell
$ pip install lazy_table
```

Examples and detailed instructions are available on the [GitHub project page](https://github.com/parsiad/lazy-table).