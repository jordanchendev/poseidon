FROM node:22-bookworm-slim AS node
FROM poseidon-closeout-test:20260921
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && npm install -g @deepseek-ai/dsh@0.1.6-alpha.2
ENV PYTHONPATH=/sdk:/workspace/scripts PYTHONDONTWRITEBYTECODE=1
WORKDIR /run-output
