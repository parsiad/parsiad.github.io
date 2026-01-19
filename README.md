My [Zola](https://getzola.org) blog using Jupytext for blog posts.

## Dependencies

* `jupytext`
* `jupyter nbconvert`
* Optional: `act`
* Optional: `watchexec`

## Usage

Create blog posts by making Jupytext files in `src/`.

You can make a debug server (needs `watchexec`) by running

```shell
./serve
```

## Publish

GitHub actions are used to publish the website on push.

Run the workflow locally (needs `act`) with

```shell
act -s GITHUB_TOKEN=$GITHUB_TOKEN
```
