# Read Me

## Function

This NeuralNetwork has 7 input neurons, 10 hidden neurons and 4 output neurons.

### Input Neurons

The input neurons are the following

|   | Empfehlung | Motivation | Fachkompetenz | Bewertung BBV | Mathe | Englisch | Deutsch |
|---|---|---|---|---|---|---|
| Inputrange | 0 - 1 | 0 - 1 | 0 - 1 | 0 - 1 | 0 - 1 |  1 - 6 |  1 - 6 |  1 - 6 |  
| Values | 1 = "++" <br> 0.666 = "+" <br> 0.333 = "+/-" <br> 0 = "-" |  1 = "++" <br> 0.666 = "+" <br> 0.333 = "+/-" <br> 0 = "-" |  1 = "++" <br> 0.666 = "+" <br> 0.333 = "+/-" <br> 0 = "-" | 1 = "TOP" <br> 0.666 = "A" <br> 0.333 = "B" <br> 0 = "C" | Schoolmarks |  Schoolmarks  Schoolmarks |

### Output Neurons

The output neurons represent the probability of the Bewertung LA1

|  | TOP | A | B | C |
|---|---|---|---|---|
| Outputrange | 0 - 1 | 0 - 1 | 0 - 1 | 0 - 1 |
| Probability | 0 to 100% | 0 to 100% | 0 to 100% | 0 to 100% |
| Example | 0.000 | 0.001 | 0.999 | 0.000 |
| Explanation | 0 % Probability that the Bewertung LA1 is "TOP" |  0.1 % Probability that the Bewertung LA1 is "A" | 99.9 % Probability that the Bewertung LA1 is "B" |  0 % Probability that the Bewertung LA1 is "C" |