* docker buildx build --platform linux/amd64 . -f Dockerfile.runpod --cache-to=type=local,dest=.buildx_cache --cache-from=type=local,src=.buildx_cache --push -t ryurchik/rvc-runtime:v1 --target build 
* docker buildx build --platform linux/amd64 . -f Dockerfile.production --load --cache-to=type=local,dest=.buildx_cache --cache-from=type=local,src=.buildx_cache 
