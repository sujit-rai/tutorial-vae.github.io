---
layout: default
---
This is a preliminary report about the tutorial on Variational AutoEncoder. This report consists of short summary about the concepts and experiments that will be discussed in the tutorial.

# Outline
- Basics of autoencoders
- Generative models
- Intuition behind Variational autoencoder
- Maths behind Variational autoencoder
- Experiments 
  - Latent Space Visualizations
  - Visualization of cluster formation
  - Effect of change in weightage for KL divergance during training
  - Effect of weightage of KL divergance on disentangled representation learning
  - Shortcoming of VAE
- Applications of VAE

## Autoencoders

These are machine learning models under unsupervised learning that come with a goal to learn good representations by trying to reconstruct the input itself. Main problem of autoencoders is not let it learn a identity function which is alleviated by regularized autoencoders(eg Sparse Autoencoders). Sparse Autoencoders come with a motive of getting sparse representations in latent space which essentially means that only few neurons are active for a particular data point. This sparse constraint in latent space forces the model to learn more good representations. In regularized autoencoders, we actually misuse the meaning of regularization. By definition regularization is our prior belief on distribution of model’s parameters where as in regularized autoencoders the regularization is a prior assumed on latent space which is **not on parameters rather on data.**






> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

### Header 3

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
