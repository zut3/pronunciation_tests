import aiosqlite as sql


async def _get_db():
    return await sql.connect('./db.sqlite')

async def _get_cursor(*args, **kwargs):
    db = await _get_db()
    return await db.execute(*args, **kwargs)

async def get_random():
    sq =  'SELECT * FROM texts ORDER BY RANDOM() LIMIT 1'
    cursor = await _get_cursor(sq)
    row_ = await cursor.fetchone()
    await cursor.close()
    return row_
    
