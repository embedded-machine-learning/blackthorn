## Benchmark Layer Configuration

Create a template dictionary for layer configuration using
```python
dict.fromkeys(<Enum Class>)
```
The available choices for `<Enum Class>` can be found [here](../layers/layer_definitions.py) 

### Layer Configuration Notes

#### Convolution
| Parameter   | Explanation                         | Details                                        |
| :---------: | :---------------------------------: | :--------------------------------------------: |
| W_IN, H_IN  | Width and height of the input       |                                                |
| D_IN, D_OUT | Number of input and output channels |                                                |
| SIZE        | Size of the convolution kernel      | Square kernels only (SIZE <dl>&times<dl> SIZE) |
| STRIDE      | Stride                              |                                                |
| PAD         | Padding                             | 'auto' to keep the input size<sup>1</sup> or any Integer  |



<sup>1</sup> Automode calculates padding using: `Int(SIZE/2)`
