import { ApolloServer } from "@apollo/server"
import { ApolloServerPluginDrainHttpServer } from "@apollo/server/plugin/drainHttpServer"
import { expressMiddleware } from "@as-integrations/express5"
import { createServer } from "http"
import { makeExecutableSchema } from "@graphql-tools/schema"
import { WebSocketServer } from "ws"
import { useServer } from "graphql-ws/use/ws"
import { PubSub } from "graphql-subscriptions"
import { readFileSync } from "fs"
import express from "express"
import cors from "cors"

// A schema is a collection of type definitions (hence "typeDefs")
// that together define the "shape" of queries that are executed against
// your data.
const typeDefs = readFileSync("../schema/weather.graphql", "utf8")

const weatherData = {
  GN: [
    {
      imageQuality: 1.0,
      cloudCover: 1.0,
      windDirection: 0.0,
      windSpeed: 0.0,
    },
  ],
  GS: [
    {
      imageQuality: 1.0,
      cloudCover: 1.0,
      windDirection: 0.0,
      windSpeed: 0.0,
    },
  ],
}

const pubsub = new PubSub()

interface Weather {
  imageQuality: number
  cloudCover: number
  windDirection: number
  windSpeed: number
  site: Site
}

type Site = "GN" | "GS"

// Resolvers define the technique for fetching the types defined in the
// schema. This resolver retrieves weather data from the "weatherData" array above.
const resolvers = {
  Query: {
    weather: () => {
      const lastWeatherBySite = Object.keys(weatherData).map((site) => ({
        site,
        ...weatherData[site as Site][weatherData[site as Site].length - 1],
      }))
      console.log("Last Weather status by site query requested:")
      console.log(lastWeatherBySite)
      return lastWeatherBySite
    },
  },

  Mutation: {
    updateWeather: (
      _: null | undefined,
      { weatherInput }: { weatherInput: Weather },
    ) => {
      console.log("Update weather mutation received:")
      console.log(weatherInput)
      const site = weatherInput.site as Site
      weatherData[site].push({
        imageQuality: weatherInput.imageQuality,
        cloudCover: weatherInput.cloudCover,
        windDirection: weatherInput.windDirection,
        windSpeed: weatherInput.windSpeed,
      })
      pubsub.publish("WEATHER_UPDATES", {
        weatherUpdates: {
          site,
          ...weatherData[site][weatherData[site].length - 1],
        },
      })
      return { site, ...weatherData[site][weatherData[site].length - 1] }
    },
  },

  Subscription: {
    weatherUpdates: {
      // Example using an async generator
      subscribe: () => {
        return pubsub.asyncIterableIterator("WEATHER_UPDATES")
      },
    },
  },
}

// Required logic for integrating with Express
const app = express()
// Our httpServer handles incoming requests to our Express app.
// Below, we tell Apollo Server to "drain" this httpServer,
// enabling our servers to shut down gracefully.
const httpServer = createServer(app)

const schema = makeExecutableSchema({ typeDefs, resolvers })

// Creating the WebSocket server
const wsServer = new WebSocketServer({
  // This is the `httpServer` we created in a previous step.
  server: httpServer,
  // Pass a different path here if app.use
  // serves expressMiddleware at a different path
  path: "/",
})
// Hand in the schema we just created and have the
// WebSocketServer start listening.
const serverCleanup = useServer({ schema }, wsServer)

const server = new ApolloServer({
  schema,
  plugins: [
    // Proper shutdown for the HTTP server.
    ApolloServerPluginDrainHttpServer({ httpServer }),
    // Proper shutdown for the WebSocket server.
    {
      async serverWillStart() {
        return {
          async drainServer() {
            await serverCleanup.dispose()
          },
        }
      },
    },
  ],
})

// Ensure we wait for our server to start
await server.start()

// Set up our Express middleware to handle CORS, body parsing,
// and our expressMiddleware function.
app.use(
  "/",
  cors<cors.CorsRequest>(),
  express.json(),
  // expressMiddleware accepts the same arguments:
  // an Apollo Server instance and optional configuration options
  expressMiddleware(server, {
    context: async ({ req }) => ({ token: req.headers.token }),
  }),
)

const PORT = process.env.PORT || 4000
// Now that our HTTP server is fully set up, we can listen to it.
httpServer.listen(PORT, () => {
  console.log(`Server is now running on http://localhost:${PORT}`)
})
