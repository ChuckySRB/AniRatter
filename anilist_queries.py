def animelist_query(username, status, type):


    return '''query {
  MediaListCollection(userName:"''' + username + '''", status''' + status + ''', type:''' + type + '''){
  	lists{
      entries{
        media{
          title{
            romaji
          }
          type
          format
          startDate {
            year
          }
          source
          genres
          meanScore
          averageScore
          popularity
          favourites
          tags{
            rank
            name
          }
        }
        score(format:POINT_100)
      }
    }

  }
}
'''
