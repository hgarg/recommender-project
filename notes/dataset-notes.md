# dataset notes

Numbers from my earlier EDA pass (back when I was still figuring out the topic), keeping
them here so they don't get lost between assignments.

## raw

- ~67,967 interactions
- 5,000 users
- 1,988 products
- sparsity: 99.32% (basically what you'd expect from this kind of interaction data,
  most users only touch a handful of products)

## after filtering out low-activity users/items

- 56,417 interactions
- 1,754 users
- 756 products
- sparsity: 95.75%

filtering approach = stratified, not just a blanket cutoff. bucketed users and items by
how much activity they had, sampled proportionally from each bucket instead of dropping
everything under some arbitrary threshold. wanted to keep some low-activity users/items
around on purpose since that's realistic and matters for the cold-start part of the
project.

## other stuff I found poking around

- 72.2% of ratings are 4 or 5 stars - pretty strong positivity bias, need to weight
  interactions instead of trusting raw star count for anything downstream
- top 5% of products account for ~47.2% of all interactions - long tail problem, a plain
  popularity-based model would basically just recommend the same 30-40 products over and
  over
- built a TF-IDF matrix off product category + price bucket + avg rating -> shape came
  out to (756, 14). feature count is low because the content field is short (basically 3
  things concatenated), might revisit if I get access to actual product descriptions

## not done yet

- haven't actually written the script that reproduces this from scratch, still mostly
  did this in a notebook the first time around and need to formalize it
- need to pick and lock in the activity filter thresholds (what counted as "low
  activity") instead of the informal cutoff I used originally
