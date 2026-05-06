"""AniList GraphQL query for MediaListCollection."""

MEDIA_LIST_QUERY = """
query ($userName: String, $type: MediaType, $statusIn: [MediaListStatus], $statusNotIn: [MediaListStatus]) {
  MediaListCollection(userName: $userName, type: $type, status_in: $statusIn, status_not_in: $statusNotIn) {
    lists {
      entries {
        score(format: POINT_100)
        media {
          id
          title { romaji english native }
          type
          format
          startDate { year }
          source
          genres
          meanScore
          averageScore
          popularity
          favourites
          episodes
          chapters
          coverImage { large }
          siteUrl
          tags { rank name }
        }
      }
    }
  }
}
""".strip()


def variables_rated(user_name: str, media_type: str) -> dict:
    """Variables to fetch a user's rated (non-planning) list of a given media type."""
    return {
        "userName": user_name,
        "type": media_type,
        "statusNotIn": ["PLANNING"],
    }


def variables_planning(user_name: str, media_type: str) -> dict:
    """Variables to fetch a user's plan-to-watch/read list of a given media type."""
    return {
        "userName": user_name,
        "type": media_type,
        "statusIn": ["PLANNING"],
    }
