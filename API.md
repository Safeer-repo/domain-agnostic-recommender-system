# API Documentation

_Generated on 2025-05-11 10:36:16_

## `POST /token`

**Description:** No description available.

---

## `POST /user/create`

**Description:** Create a new user account

**Request Body Example:**
```json
{
  "username": "sample_username",
  "email": null,
  "password": "sample_password",
  "domain_preferences": null
}
```
---

## `POST /user/login`

**Description:** Login a user and return an access token

**Request Body Example:**
```json
{
  "username": "sample_username",
  "password": "sample_password"
}
```
---

## `GET /user/me`

**Description:** Get the current user's profile information

---

## `POST /user/rate`

**Description:** Submit a new rating for an item

**Request Body Example:**
```json
{
  "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
  "item_id": 12345,
  "rating": 4.5,
  "source": "explicit",
  "domain": "entertainment",
  "dataset": "movielens"
}
```
---

## `POST /user/rate/batch`

**Description:** Submit multiple ratings at once

**Request Body Example:**
```json
{
  "domain": "entertainment",
  "dataset": "movielens",
  "ratings": [
    {
      "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
      "item_id": 12345,
      "rating": 4.5,
      "source": "explicit"
    },
    {
      "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
      "item_id": 67890,
      "rating": 3.0,
      "source": "explicit"
    }
  ]
}
```
---

## `POST /user/feedback`

**Description:** Record user interaction with a recommendation

**Request Body Example:**
```json
{
  "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
  "item_id": 12345,
  "interaction_type": "click",
  "timestamp": 1620000000,
  "domain": "entertainment",
  "dataset": "movielens"
}
```
---

## `POST /recommendations/user`

**Description:** Get personalized recommendations for a user

**Request Body Example:**
```json
{
  "user_id": "c210ca32-ad7c-41a5-a806-71c95dfce5ec",
  "domain": "entertainment",
  "dataset": "movielens",
  "count": 10,
  "filters": null
}
```
---

## `POST /recommendations/similar`

**Description:** Get items similar to a given item

**Request Body Example:**
```json
{
  "item_id": 12345,
  "domain": "entertainment",
  "dataset": "movielens",
  "count": 5
}
```
---

## `GET /models/{domain}/{dataset}`

**Description:** Get information about the currently active model for a domain/dataset

**Path Parameters:**
- `domain`
- `dataset`

---

## `GET /models/{domain}/{dataset}/history`

**Description:** Get the training history for models in a domain/dataset

**Path Parameters:**
- `domain`
- `dataset`

---

## `POST /admin/update-dataset`

**Description:** Trigger the dataset update process to incorporate new ratings

**Request Body Example:**
```json
{
  "domain": "sample_domain",
  "dataset": "sample_dataset",
  "incremental": null
}
```
---

## `POST /admin/retrain`

**Description:** Trigger model retraining for a domain/dataset

**Request Body Example:**
```json
{
  "domain": "sample_domain",
  "dataset": "sample_dataset",
  "algorithm": null,
  "force": null
}
```
---

## `GET /admin/stats`

**Description:** Get statistics about the recommender system

---

## `GET /health`

**Description:** Health check endpoint for monitoring

---

## `GET /`

**Description:** API root endpoint with basic information

---
