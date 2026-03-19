const express = require('express')
const sql = require('mssql')
const cors = require('cors')
const { spawn } = require('child_process')
const path = require('path')

const app = express()
app.use(cors())

const config = {
  user: 'almighty',
  password: 'almighty',
  server: '127.0.0.1',
  port: 22020,
  database: 'SmartdoctorDB',
  options: {
    encrypt: false,
    trustServerCertificate: true
  }
}

// ✅ AI 실행 API (🔥 최종 완성)
app.get('/run-ai', async (req, res) => {
  try {
    const pool = await sql.connect(config)

    // 1️⃣ 환자 조회
    const result = await pool.request().query(`
      SELECT 
        CUST_NO,
        ISNULL(avg_gap_days, 0) as gap_days,
        ISNULL(visit_count, 0) as visit_count,
        ISNULL(days_since_last_visit, 0) as days_since_visit
      FROM patient_summary
    `)

    const patients = result.recordset
    console.log(`총 환자 수: ${patients.length}`)

    // 2️⃣ Python batch 실행
    const pythonFile = path.join(__dirname, '..', 'ai', 'predict_batch.py')

    const py = spawn('python', [pythonFile])

    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (data) => {
      stdout += data.toString()
    })

    py.stderr.on('data', (data) => {
      stderr += data.toString()
    })

    py.stdin.write(JSON.stringify(patients))
    py.stdin.end()

    py.on('close', async (code) => {
      if (code !== 0) {
        console.error('❌ Python 에러:', stderr)
        return res.status(500).send(stderr)
      }

      let predictions
      try {
        predictions = JSON.parse(stdout)
      } catch (e) {
        console.error('❌ JSON 파싱 실패:', stdout)
        return res.status(500).send('JSON parse error')
      }

      console.log(`예측 완료: ${predictions.length}`)

      // 🔥 3️⃣ 일반 테이블 생성 (temp X)
      await pool.request().query(`
        IF OBJECT_ID('temp_predictions', 'U') IS NOT NULL
          DROP TABLE temp_predictions

        CREATE TABLE temp_predictions (
          CUST_NO NVARCHAR(50),
          revisit_prob FLOAT
        )
      `)

      // 🔥 4️⃣ bulk insert
      const table = new sql.Table('temp_predictions')
      table.columns.add('CUST_NO', sql.NVarChar(50))
      table.columns.add('revisit_prob', sql.Float)

      predictions.forEach(p => {
        table.rows.add(p.CUST_NO, p.revisit_prob)
      })

      await pool.request().bulk(table)

      // 🔥 5️⃣ 한번에 UPDATE
      await pool.request().query(`
        UPDATE p
        SET p.revisit_prob = t.revisit_prob
        FROM patient_summary p
        JOIN temp_predictions t
        ON p.CUST_NO = t.CUST_NO
      `)

      // 🔥 6️⃣ 테이블 삭제
      await pool.request().query(`DROP TABLE temp_predictions`)

      console.log('🔥 DB 업데이트 완료')

      res.send('🔥 AI 배치 업데이트 완료')

    })

  } catch (err) {
    console.error('❌ 서버 에러:', err)
    res.status(500).send(err.message || 'Error')
  }
})

// ✅ 프론트용 API
app.get('/patient-risk', async (req, res) => {
  try {
    const pool = await sql.connect(config)

    const result = await pool.request().query(`
      SELECT *,
      CASE 
        WHEN revisit_prob >= 0.7 THEN 'HIGH'
        WHEN revisit_prob >= 0.4 THEN 'MID'
        ELSE 'LOW'
      END as risk_level_new
      FROM patient_summary
    `)

    res.json(result.recordset)

  } catch (err) {
    console.error(err)
    res.status(500).send('Error')
  }
})

app.listen(3000, () => {
  console.log('🔥 Server running on 3000')
})