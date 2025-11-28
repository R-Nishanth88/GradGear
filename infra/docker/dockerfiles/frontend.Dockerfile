FROM node:18-alpine
WORKDIR /app
COPY apps/frontend/package.json ./package.json
RUN npm install --legacy-peer-deps || true
COPY apps/frontend /app
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]


