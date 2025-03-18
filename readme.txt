在仿真环境simpy 中 必须保证时间是按照整数来走的 不然 容易出现小数点

关键点：
        cmt = np.around(ptl * mpc, 2)
        cmt = np.rint(ptl * mpc)
    改成整数

    1. WINQ（Work in Next Queue，下一队列的工作量）
    2. AVLM（Available Machine Time，可用机器时间）