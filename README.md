# Beyond the Obsessed Encoder

**Let control shape the latent space of visual world models.**

This personal repository hosts an [illustrated research post](index.html) about a predictable visual nuisance in Tagged PushT and an action-derived gradient-routing pilot. The work is **independent research inspired by** Enigma Incorporated's [The Obsessed Encoder](https://www.enigma.inc/posts/obsessed-encoder). The authors are not members of Enigma Incorporated, and this is not an official continuation of its post.

The post links three kinds of evidence: latent allocation, fixed-candidate planner interventions, and closed-loop CEM success. Its reported control results use **one training seed** and **50 evaluation episodes per task**. The spatial sensitivity heatmap is an illustrative matched scene; the fixed-candidate intervention rates are aggregate diagnostics. This is a research preview, not a multi-seed benchmark claim.

## Read and inspect

- [Read the post](index.html)
- [Read the Markdown source](article.md)
- [View the presentation](assets/project-presentation.pdf)
- [Inspect the development implementation branch](https://github.com/Enigma-Incorporated/The-Obsessed-Encoder/tree/experiment/randgoal-decision-aligned), currently hosted in the original repository

The method discussed here is the **parameter-space EMA control-gradient router** from the saved October 2026 presentation. Later experimental variants are separate and their results are not pooled into the post.

## Website

The site is plain HTML and CSS. `index.html` and `assets/` can be opened locally or published directly through GitHub Pages from the repository's `main` branch and `/ (root)` folder. `.nojekyll` keeps the generated site unchanged.

## Authors

Changchen Song and Jiahang Cao.
