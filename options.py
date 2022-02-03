from scipy.stats import norm, mvn
import numpy as np
import datetime

def black_scholes(call_put_flag, s, x, t, r, v):
    d1 = (np.log(np.divide(s, x)) + (r + np.power(v, 2) / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)

    if call_put_flag == "c":
        return s * norm.cdf(d1) - x * np.exp(-r * t) * norm.cdf(d2)
    elif call_put_flag == 'p':
        return x * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


#call_price = black_scholes("c", 60, 65, 0.25, 0.08, 0.3)


def GBlackScholes(call_put_flag, s, x, t, r, b, v):
    d1 = (np.log(np.divide(s, x)) + (b + np.power(v, 2) / 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)

    if call_put_flag == "c":
        return s * np.exp(np.multiply(np.subtract(b, r), t)) * norm.cdf(d1) - x * np.exp(np.multiply(-r, t)) * norm.cdf(d2)
    elif call_put_flag == 'p':
        return x * np.exp(np.multiply(-r, t)) * norm.cdf(-d2) - s * np.exp(np.multiply(np.subtract(b, r), t)) * norm.cdf(-d1)

#put_price = GBlackScholes("p", 75, 70, 0.5, 0.1, 0.05, 0.35)


def kc(x, t, r, b, v):
    # Calculation of seed value
    N = 2 * b / np.power(v, 2)
    m = 2 * r / np.power(v, 2)
    q2u = (-(N-1) + np.sqrt(np.power(np.subtract(N, 1), 2)) + 4 * m) / 2
    su = x / (1 - 1/ q2u)
    h2 = -(b * t + 2 * v * np.sqrt(t) * x / (su - x))
    si = x + (su - x) * (1 - np.exp(h2))

    k = 2 * r / (np.power(v, 2) * (np.subtract(1, np.exp(np.multiply(-r, t)))))
    d1 = (np.log(np.divide(si, x)) + (b + np.power(v, 2) / 2) * t) / (np.multiply(v, np.sqrt(t)))

    q2 = (-(N - 1) + np.sqrt(np.power(N - 1, 2) + 4 * k)) / 2
    lhs = si - x
    rhs = GBlackScholes("c", si, x, t, r, b, v) + (1 - np.exp(np.multiply(b - r, t)) * norm.cdf(d1)) * np.divide(si, q2)
    bi = np.exp(np.multiply(b - r, t)) * norm.cdf(d1) * (1 - 1 / q2) + (1 - np.exp(np.multiply(b - r, t)) * norm.cdf(d1) / np.multiply(v, np.sqrt(t))) / q2

    e = 1e-6

    while np.abs(lhs, rhs) / x > e:
        si = (x + rhs - bi * si) / (1 - bi)
        d1 = (np.log(np.divide(si, x)) + (b + np.power(v, 2) / 2) * t) / np.multiply(v, np.sqrt(t))
        lhs = si - x
        rhs = GBlackScholes('c', si, x, t, r, b, v) + (1 - np.exp(np.multiply(b - r, t)) * norm.cdf(d1)) * si / q2
        bi = np.exp(np.multiply(b - r, t)) * norm.cdf(d1) * (1 - 1 / q2) + (1 - np.exp(np.multiply(b -r, t)) * norm.cdf(d1) / (v * np.sqrt(t))) / q2

    return si

def BAWAAmericanCallAprox(s, x, t, r, b, v):
    if b >= r:
        return GBlackScholes("c", s, x, t, r, b, v)
    else:
        sk = kc(x, t, r, b, v)
        n = 2 * b / np.power(v, 2)
        k = 2 * r / (np.power(v, 2) * (1- np.exp(-r * t)))
        d1 = (np.log(sk / x) + (b + np.power(v, 2) / 2) * t) / (np.multiply(v, np.sqrt(t)))
        q2 = (-(n-1) + np.sqrt(np.power(n-1, 2) + 4 * k)) / 2
        a2 = (sk / q2) * (1 - np.exp(np.multiply(b - r, t))) * norm.cdf(d1)

        if s < sk:
            return GBlackScholes("c", s, x, t, r, b, v) + a2 * np.power(np.divide(s, sk), q2)
        else:
            return s - x


#var = BAWAAmericanCallAprox(90, 100, 0.1, 0.1, 0, 0.15)

#print(var)

def phi(s, t, gamma, h, i, r, b, v):
    lamb = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * np.power(v, 2)) * t
    d1 = -(np.log(s/h) + (b + (gamma - 0.5) * np.power(v, 2)) * t) / (v * np.sqrt(t))
    d2 = d1 - 2 * np.log(i/s) / (v * np.sqrt(t))
    kappa = 2 * b / np.power(v, 2) + (2 * gamma - 1)

    return np.exp(lamb) * np.power(s, gamma) * (norm.cdf(d1) - np.power(i/s, kappa) * norm.cdf(d2))

def BSAmericanCallApprox(s, x, t, r, b, v):

    if b >= r:
        return GBlackScholes('c', s, x, t, r, b, v)
    else:
        beta = (1 / 2 - b / np.power(v, 2)) + np.sqrt(np.power(b / np.power(v, 2) - 1 / 2, 2) + 2 * r / np.power(v, 2))
        b_infinity = beta / (beta - 1) * x
        b0 = np.maximum(x, r / (r - b) * x)
        ht = -(b * t + 2 * v * np.sqrt(t)) * b0 / (b_infinity - b0)
        i = b0 + (b_infinity - b0) * (1 - np.exp(ht))
        alpha = (i - x) * np.power(i, -beta)

        #print("Beta: ", beta, " * BetaInfinity: ",  b_infinity, " * B0: ", b0, " * H(T): ", ht, " * I: ", i, " * Alpha: ", alpha)

        if s >= i:
            return s - x
        else:
            return alpha * np.power(s, beta) - alpha * phi(s, t, beta, i, i, r, b, v) + phi(s, t, 1, i, i, r, b, v) - phi(s, t, 1, x, i, r, b, v) - x * phi(s, t, 0, i, i, r, b, v) + x * phi(s, t, 0, x, i, r, b, v)


#var = BSAmericanCallApprox(42, 40, 0.75, 0.04, -0.04, 0.35)
#print(var)

def _cbnd(a, b, rho):
    lower = np.array([0, 0])
    upper = np.array([a, b])
    infin = np.array([0, 0])
    correl = rho

    error, value, inform = mvn.mvndst(lower, upper, infin, correl)
    return value

def ksi(s, t2, gamma, h, i2, i1, t1, r, b, v):
    e1 = (np.log(s / i1) + (b + (gamma - 0.5) * np.power(v, 2)) * t1) / (v * np.sqrt(t1))
    e2 = (np.log(np.power(i2, 2) / (s * i1)) + (b + (gamma - 0.5) * np.power(v, 2)) * t1) / (v * np.sqrt(t1))
    e3 = (np.log(s / i1) - (b + (gamma - 0.5) * np.power(v, 2)) * t1) / (v * np.sqrt(t1))
    e4 = (np.log(np.power(i2, 2) / (s * i1)) - (b + (gamma - 0.5) * np.power(v, 2)) * t1) / (v * np.sqrt(t1))

    f1 = (np.log(s / h) + (b + (gamma - 0.5) * np.power(v, 2)) * t2) / (v * np.sqrt(t2))
    f2 = (np.log(np.power(i2, 2) / (s * h)) + (b + (gamma - 0.5) * np.power(v, 2)) * t2) / (v * np.sqrt(t2))
    f3 = (np.log(np.power(i1, 2) / (s * h)) + (b + (gamma - 0.5) * np.power(v, 2)) * t2) / (v * np.sqrt(t2))
    f4 = (np.log((s * np.power(i1, 2)) / (h * np.power(i2, 2))) + (b + (gamma - 0.5) * np.power(v, 2)) * t2) / (v * np.sqrt(t2))

    rho = np.sqrt(t1 / t2)
    lamb = -r + gamma * b + 0.5 * gamma * (gamma - 1) * np.power(v, 2)

    #kappa = 2 * s * b / np.power(v, 2) + (2 * gamma - 1) # Original Line
    kappa = (2 * b) / np.power(v, 2) + (2 * gamma - 1)

    return np.exp(lamb * t2) * np.power(s, gamma) * (_cbnd(-e1, -f1, rho) - np.power(i2 / s, kappa) * _cbnd(-e2, -f2, rho) - \
            np.power(i1 / s, kappa) * _cbnd(-e3, -f3, -rho) + np.power(i1 / i2, kappa) * _cbnd(-e4, -f4, -rho))


def BSAmericanCallApprox2002(s, x, t, r, b, v):

    if b >= r:
        return GBlackScholes('c', s, x, t, r, b, v)

    t1 = 1 / 2 * (np.sqrt(5) - 1) * t
    beta = (1/2 - b / np.power(v, 2)) + np.sqrt(np.power(b / np.power(v, 2) - 1/2, 2) + 2 * r / np.power(v, 2))
    b_infinity = beta / (beta - 1) * x
    b0 = np.maximum(x, r / (r - b) * x)

    #ht1 = -(b * t1 + 2 * v * np.sqrt(t1)) * np.power(x, 2) / ((b_infinity - b0) * b0) # Original Line
    ht1 = -(b * (t - t1) + 2 * v * np.sqrt(t - t1)) * np.power(x, 2) / ((b_infinity - b0) * b0)
    ht2 = -(b * t + 2 * v * np.sqrt(t)) * np.power(x, 2) / ((b_infinity - b0) * b0)
    i1 = b0 + (b_infinity - b0) * (1 - np.exp(ht1))
    i2 = b0 + (b_infinity - b0) * (1 - np.exp(ht2))
    alpha1 = (i1 - x) * np.power(i1, -beta)
    alpha2 = (i2 - x) * np.power(i2, -beta)

    if s >= i2:
        value = s - x
    else:
        value = alpha2 * np.power(s, beta) - alpha2 * phi(s, t1, beta, i2, i2, r, b, v) + phi(s, t1, 1, i2, i2, r, b, v) - phi(s, t1, 1, i1, i2, r, b, v) - x * phi(s, t1, 0, i2, i2, r, b, v) + \
               x * phi(s, t1, 0, i1, i2, r, b, v) + alpha1 * phi(s, t1, beta, i1, i2, r, b, v) - alpha1 * ksi(s, t, beta, i1, i2, i1, t1, r, b, v) + ksi(s, t, 1, i1, i2, i1, t1, r, b, v) - \
                ksi(s, t, 1, x, i2, i1, t1, r, b, v) - x * ksi(s, t, 0, i1, i2, i1, t1, r, b, v) + x * ksi(s, t, 0, x, i2, i1, t1, r, b, v)

    return value

"""
###################### Parameters ##############################
s: Stock Price
x: Strike price of option
r: Risk-free interest rate
t: Time to expiration in years
b: 
v: 
"""
def BSAmericanApprox2002(call_put_flag, s, x, t, r, b, v):
    if call_put_flag == 'c':
        return BSAmericanCallApprox2002(s, x, t, r, b, v)
    elif call_put_flag == 'p':
        return BSAmericanCallApprox2002(x, s, t, r - b, -b, v)


contracts = 1153
prime = 0.74
inversion = prime * contracts * 100
print("Option Prime: ", prime)
print("Inversion: {:,.2f}".format(inversion))
print("#######################################################################")

maturity = (datetime.date(2023, 1, 19) - datetime.date.today()).days / 365
valuacion = BSAmericanApprox2002('c', 365, 570, maturity, 0.01, 0, 0.1994)
inversion_c1 = valuacion * contracts * 100
print("Option Prime", valuacion)
print("Retorno: ${:,.2f}".format(inversion_c1 - inversion))
print("#######################################################################")

maturity = (datetime.date(2023, 1, 19) - datetime.date(2022, 3, 2)).days / 365
valuacion = BSAmericanApprox2002('c', 420, 570, maturity, 0.01, 0, 0.23)
inversion_c2 = valuacion * contracts * 100
print("Option Prime", valuacion)
print("Retorno: ${:,.2f}".format(inversion_c2 - inversion))

"""
var = BSAmericanCallApprox(42, 40, 0.75, 0.04, -0.04, 0.35)
print(var)


var = BSAmericanCallApprox(80, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
print("******************************")

var = BSAmericanApprox2002('c', 80, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('c', 90, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('c', 100, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('c', 110, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('c', 120, 100, 0.25, 0.08, -0.04, 0.2)
print(var)

print("******************************")

var = BSAmericanApprox2002('p', 80, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('p', 90, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('p', 100, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('p', 110, 100, 0.25, 0.08, -0.04, 0.2)
print(var)
var = BSAmericanApprox2002('p', 120, 100, 0.25, 0.08, -0.04, 0.2)
print(var)

print("******************************")
var = BSAmericanApprox2002('c', 28.884999, 42.5, 763/365, 0.01, 0, 0.3743)
print(var)
"""