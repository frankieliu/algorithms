
# 113 and 10**9+7
if p and M are relative prime, and M is prime
then p^-1 = p^(M-2) mod M



If p and M are relatively prime (meaning their greatest common divisor is 1), the modular multiplicative inverse of p modulo M can be expressed using a positive exponent

. 

**Using Euler's Totient Theorem:** 

- Euler's Totient Theorem states that if p and M are coprime, then 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{\varphi \left(\right. M \left.\right)} \equiv 1 \left(\right. mod M \left.\right)$
    
    𝑝𝜑(𝑀)≡1(mod𝑀)
    
    , where 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $\varphi \left(\right. M \left.\right)$
    
    𝜑(𝑀)
    
    is Euler's totient function (which counts the number of positive integers less than M that are relatively prime to M).
- Multiplying both sides by 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{-1}$
    
    𝑝−1
    
    results in 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{\varphi \left(\right. M \left.\right) - 1} \equiv p^{-1} \left(\right. mod M \left.\right)$
    
    𝑝𝜑(𝑀)−1≡𝑝−1(mod𝑀)
    
    .
- Therefore, the modular inverse of p modulo M can be expressed as 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{\varphi \left(\right. M \left.\right) - 1} \left(\right. mod M \left.\right)$
    
    𝑝𝜑(𝑀)−1(mod𝑀)
    
    . This expression involves a positive exponent (
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $\varphi \left(\right. M \left.\right) - 1$
    
    𝜑(𝑀)−1
    
    ). 

**Special Case: When M is a Prime Number:** 

- If M is a prime number, then 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $\varphi \left(\right. M \left.\right) = M - 1$
    
    𝜑(𝑀)\=𝑀−1
    
    .
- In this case, Fermat's Little Theorem applies, which is a special case of Euler's Theorem.
- It states that if M is prime and p is not divisible by M, then 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{M - 1} \equiv 1 \left(\right. mod M \left.\right)$
    
    𝑝𝑀−1≡1(mod𝑀)
    
    .
- Multiplying both sides by 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{-1}$
    
    𝑝−1
    
    results in 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{M - 2} \equiv p^{-1} \left(\right. mod M \left.\right)$
    
    𝑝𝑀−2≡𝑝−1(mod𝑀)
    
    .
- So, if M is prime, the modular inverse of p modulo M is 
    
    ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{M - 2} \left(\right. mod M \left.\right)$
    
    𝑝𝑀−2(mod𝑀)
    
    , again expressed with a positive exponent. 

**In summary:** 

If p and M are relatively prime, the modular inverse of p modulo M (

![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)

$p^{-1} \left(\right. mod M \left.\right)$

𝑝−1(mod𝑀)

) can be represented as: 

- ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{\varphi \left(\right. M \left.\right) - 1} \left(\right. mod M \left.\right)$
    
    𝑝𝜑(𝑀)−1(mod𝑀)
    
    (using Euler's Totient Theorem)
- ![](data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==)
    
    $p^{M - 2} \left(\right. mod M \left.\right)$
    
    𝑝𝑀−2(mod𝑀)
    
    (if M is prime, using Fermat's Little Theorem) 

These expressions use a positive exponent to define the modular inverse.