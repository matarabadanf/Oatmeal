import numpy as np 
from numpy.typing import NDArray
import matplotlib.pyplot as plt 

def even_tempered_exponents(alpha_1: float, epsilon:float, k: int) -> NDArray[np.float64]:

    exponents = np.zeros(k)

    for i in range(k):
        exponents[i] = alpha_1 * epsilon ** (i-1)

    return exponents

def even_tempered_demonstration(alpha_1: float = 0.5, epsilon: float = 0.5, k: int = 10) -> None:

    exponents = even_tempered_exponents(alpha_1, epsilon, k)

    grid = np.linspace(-10, 10, 300)

    for exp in exponents:
        image = np.e**(-exp*grid**2)
        plt.plot(grid, image, label=f'a = {exp}')
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    even_tempered_demonstration()