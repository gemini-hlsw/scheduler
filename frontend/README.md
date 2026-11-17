# schedule

GPP Schedule is the app that allows QCs to interact with the (Automated Scheduler)[https://github.com/gemini-hlsw/scheduler]

## Start local development environment

### Node installation

An installation of Node.js version 24 is required.

For Linux and macOS the following steps can be used

```bash
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"
# Download and install Node.js:
nvm install 24
# Verify the Node.js version:
node -v # Should print "v24.11.1".
nvm current # Should print "v24.11.1".
# Download and install pnpm:
corepack enable pnpm
# Verify pnpm version:
pnpm -v
```

### Environmental Variables

The UI can connect to any running scheduler server, local or cloud. By default it will try to connect to the `heroku` running instance.

To connect a local server running on the port 8000 (default server value), create an `.env` file with the following content on this repository root directory.

```bash
VITE_API_URL=http://localhost:8000/graphql
```

You should be able to see the `.env` file in the same level of the `package.json` file, like in the following tree.

```
.
├── codegen.yml
├── .env
├── .gitignore
├── index.html
├── package.json
...
└── src
```

### Install web dependencies

Open a terminal in the root directory of this repository and run

```bash
pnpm install
```

### Start web server

The web server can be started using the following command in the root directory of this repository

```bash
pnpm dev
```

If everything goes well you should be able to see something similar to the following messages in your terminal

```bash
  VITE v4.5.9  ready in 366 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://172.28.71.166:5173/
  ➜  Network: http://172.20.0.1:5173/
  ➜  Network: http://172.19.0.1:5173/
  ➜  Network: http://10.91.2.1:5173/
  ➜  Network: http://10.91.8.1:5173/
  ➜  press h to show help
```

This mean now you should be able to connect to http://localhost:5173 web and see the UI running.

### Python scheduler server

If the UI is configured to connect to a local running python instance, please make sure you have a started instance of the scheduler.

You can check this link https://github.com/gemini-hlsw/scheduler?tab=readme-ov-file#service to start a local instance.

### Possible Issues

#### CORS

To check if CORS is being an issue, in the web browser you are using open the `Developer Tools` > `Console` and refresh the website. If the console displays an error message like this

```
Access to fetch at 'http://localhost:8000/graphql' from origin 'http://localhost:5173' has been blocked by CORS policy: Response to preflight request doesn't pass access control check: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

The UI will try to connect to the scheduler server using the origin provided in its configuration, by default it should be http://localhost:5173, so the server should be configured to allow this origin, it should be done by default in the latest version, this can be checked in the scheduler repository, `app.py` file, `origins` should list `http://localhost:5173` among the possible options, if it is not there pull the latest version of the repository or just add it.

Also make sure the `.env` file was created in the UI root directory using the right value for `VITE_API_URL`.
