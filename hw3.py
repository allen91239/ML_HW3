import numpy as np
import math
import matplotlib.pyplot as plt

def uni_gaussian(mean, variance):
    s = 1
    while(s >= 1):
        u = np.random.uniform(-1,1,2)
        s = u[0]**2 + u[1]**2
        if(-2*np.log(s)/s > 0 and s < 1):
            x = u[0] * np.sqrt((-2*np.log(s)/s)) * variance + mean
            data_point = x
    return data_point
        
def poly_basis(e, n, a, w):
    x = np.random.uniform(-1,1,1)
    y = 0.0
    for i in range(n):
        y = y + w[i]*x**i
    y = y + e
    return x, y

def seq_estimator(mean, variance):
    print(f"Data point source function: N({mean}, {variance})")
    m = 0.0
    v = 0.0
    M2 = 0.0
    count = 0
    while(abs(m - mean) > 0.04 or abs(v - variance) > 0.04):
        new_point = uni_gaussian(mean, np.sqrt(variance))
        count += 1
        delta = new_point - m
        m += delta/count
        delta2 = new_point - m
        M2 += delta * delta2
        v = M2 / count
        print(f"Add data point: {new_point}")
        print(f"Mean = {m} \t Variance = {v}")
def plot_graph(count, result, b, n, a, w, e, cov):
    plt_x = np.linspace(-2, 2, 200)
    plt_y = np.zeros(200)
    var = list()
    for i in range(200):
        tmp = e
        for j in range(n):
            tmp += w[j] * plt_x[i]**j
        plt_y[i] = tmp
    for i in range(200):
        x = np.array([plt_x[i]**j for j in range(n)])
        var.append(1 / a + np.dot(np.dot(x, np.linalg.inv(cov)), np.transpose(x)))
    y_var = np.array(var)
    plt.plot(plt_x, plt_y, 'k')
    plt.plot(plt_x, plt_y + y_var, 'r')
    plt.plot(plt_x, plt_y - y_var, 'r')
    plt.scatter(result[:, 0], result[:, 1])
    plt.savefig(f"Number{count}.png")
    plt.close()

def bayes_linreg():
    b = float(input("Input precision b: "))
    n = int(input("Input basis n: "))
    a = float(input("Input variance of N(0, a) a: "))
    w = np.zeros(n)
    print("Input array w: ")
    for i in range(n):
        w[i] = float(input(f"w[{i+1}]= "))

    plt_x = np.linspace(-2, 2, 200)
    plt_y = np.zeros(200)
    
    posterior_mean = np.zeros((n, 1))
    posterior_cov = np.zeros((n,n))
    prior_mean = np.zeros((n,1))
    e0 = uni_gaussian(0, np.sqrt(a))
    x0, y0 = poly_basis(e0, n, np.sqrt(a), w)
    for i in range(200):
        tmp = e0
        for j in range(n):
            tmp += w[j] * plt_x[i]**j
        plt_y[i] = tmp
    plt.plot(plt_x, plt_y, 'k')
    plt.plot(plt_x, plt_y + np.sqrt(a), 'r')
    plt.plot(plt_x, plt_y - np.sqrt(a), 'r')
    plt.savefig("ground_truth.png")
    plt.close()
    
    x0 = x0[0]
    x_point = np.array(x0)
    e_x = np.array(e0)
    X = np.array([[x0**i for i in range(n)]])
    y = np.array([[y0[0]]])
    posterior_cov = a * np.dot(np.transpose(X), X) + b * np.identity(n)
    posterior_mean = a * np.dot(np.linalg.inv(posterior_cov), np.transpose(X)) * y[0, 0]
    xp = X[0]
    predict_mean = np.dot(xp, posterior_mean)[0]
    predict_var = 1 / a + np.dot(np.dot(xp, np.linalg.inv(posterior_cov)), np.transpose(xp))

    print(f"Add data point ({x0}, {y0[0]}):")
    print()
    print("Posterior mean:")
    print(posterior_mean)
    print()
    print("Posterior variance:")
    print(posterior_cov)
    print()
    print(f"Predictive distribution ~ N({predict_mean}, {predict_var}")
    print("--------------------------------------------------------")
    predict_out = np.array([[x0, predict_mean]])
    counter = 0
    while((np.abs(posterior_mean-prior_mean)>0.00001).all() ):
        counter += 1
        prior_mean = posterior_mean
        e = uni_gaussian(0, np.sqrt(a))
        x, in_y = poly_basis(e, n, np.sqrt(a), w)
        x = x[0]
        #x_point = np.append(x_point, x, axis = 0)
        #e_x = np.append(e_x, e, axis = 0)
        S = posterior_cov
        m = posterior_mean

        X = np.append(X, np.array([[x**i for i in range(n)]]), axis = 0)
        y = np.append(y, np.array([[in_y[0]]]), axis = 0)
        posterior_cov = a * np.dot(np.transpose(X), X) + S
        posterior_mean = np.dot(np.linalg.inv(posterior_cov), a * np.dot(np.transpose(X), y) + np.dot(S, m))
        xp = X[counter]
        predict_mean = np.dot(xp, posterior_mean)[0]
        predict_var = 1 / a + np.dot(np.dot(xp, np.linalg.inv(posterior_cov)), np.transpose(xp))

        print(f"Add data point ({x}, {in_y[0]}):")
        print()
        print("Posterior mean:")
        print(posterior_mean)
        print()
        print("Posterior variance:")
        print(np.linalg.inv(posterior_cov))
        print()
        print(f"Predictive distribution ~ N({predict_mean}, {predict_var})")
        print("--------------------------------------------------------")
        predict_out = np.append(predict_out, [[x, predict_mean]], axis = 0)

        if(counter == 10):
            plot_graph(counter, predict_out, b, n, predict_var, w, e, posterior_cov)
        if(counter == 20):
            plot_graph(counter, predict_out, b, n, predict_var, w, e, posterior_cov)
        if(counter == 50):
            plot_graph(counter, predict_out, b, n, predict_var, w, e, posterior_cov)
    plt_x = np.linspace(-2, 2, 200)
    plt_y = np.zeros(200)
    for i in range(200):
        tmp = e
        for j in range(n):
            tmp += w[j] * plt_x[i]**j
        plt_y[i] = tmp
    plt.plot(plt_x, plt_y, 'k')
    plt.plot(plt_x, plt_y + np.sqrt(a), 'r')
    plt.plot(plt_x, plt_y - np.sqrt(a), 'r')
    plt.scatter(predict_out[:, 0], predict_out[:, 1])
    plt.savefig("result.png")
    plt.close()
    

if __name__ == "__main__":
    mean = float(input("Input mean: "))
    variance = float(input("Input variance: "))
    seq_estimator(mean, variance)
    #bayes_linreg()